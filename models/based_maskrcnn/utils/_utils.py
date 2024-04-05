import inspect
import functools
import warnings
from enum import Enum
from typing import Any, Dict, Optional, TypeVar, Callable, Tuple, Union

from torch import nn

from ._api import WeightsEnum


__all__ = [
    'overwrite_eps',
    'InterpolationMode',
    'handle_legacy_interface',
    '_ovewrite_named_param',
    '_ovewrite_value_param',
]


def overwrite_eps(model: nn.Module, eps: float) -> None:
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.
    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


D = TypeVar("D")


def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.
    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:
    .. code::
        def old_fn(foo, bar, baz=None):
            ...
        def new_fn(foo, *, bar, baz=None):
            ...
    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:
    .. code::
        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...
        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated. Please use keyword parameter(s) instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper


W = TypeVar("W", bound=WeightsEnum)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]):
    """Decorates a model builder with the new interface to make it compatible with the old.
    In particular this handles two things:
    1. Allows positional parameters again, but emits a deprecation warning in case they are used. See
        :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None`` and ``pretrained=True`` to
        ``weights=Weights`` and emits a deprecation warning with instructions for the new interface.
    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): Deprecated parameter
            name and default value for the legacy ``pretrained=True``. The default value can be a callable in which
            case it will be called with a dictionary of the keyword arguments. The only key that is guaranteed to be in
            the dictionary is the deprecated parameter name passed as first element in the tuple. All other parameters
            should be accessed with :meth:`~dict.get`.
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():  # type: ignore[union-attr]
                # If neither the weights nor the pretrained parameter as passed, or the weights argument already use
                # the new style arguments, there is nothing to do. Note that we cannot use `None` as sentinel for the
                # weight argument, since it is a valid value.
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)
                if (
                    (weights_param not in kwargs and pretrained_param not in kwargs)
                    or isinstance(weights_arg, WeightsEnum)
                    or (isinstance(weights_arg, str) and weights_arg != "legacy")
                    or weights_arg is None
                ):
                    continue

                # If the pretrained parameter was passed as positional argument, it is now mapped to
                # `kwargs[weights_param]`. This happens because the @kwonly_to_pos_or_kw decorator uses the current
                # signature to infer the names of positionally passed arguments and thus has no knowledge that there
                # used to be a pretrained parameter.
                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    # We put the pretrained argument under its legacy name in the keyword argument dictionary to have a
                    # unified access to the value if the default value is a callable.
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]

                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated, please use '{weights_param}' instead."
                    )

                msg = (
                    f"Arguments other than a weight enum or `None` for '{weights_param}' are deprecated. "
                    f"The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.DEFAULT` "
                        f"to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: Optional[V], new_value: V) -> V:
    if param is not None:
        if param != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {param} instead.")
    return new_value
