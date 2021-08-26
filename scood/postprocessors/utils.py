from .base_postprocessor import BasePostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .odin_postprocessor import ODINPostprocessor


def get_postprocessor(name: str, **kwargs):
    postprocessors = {
        "none": BasePostprocessor,
        "ebo": EBOPostprocessor,
        "odin": ODINPostprocessor,
    }

    return postprocessors[name](**kwargs)
