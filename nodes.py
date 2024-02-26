from random import random
from HablabaNodes.comfy_annotations import (
    NumberInput,
    ComfyFunc,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
)
import torch

my_category = "Comfy Annotation Examples"


# This is the converted example node from ComfyUI's example_node.py.example file.
@ComfyFunc(my_category)
def annotated_example(
    image: ImageTensor,
    string_field: str = StringInput("Hello World!", multiline=False),
    int_field: int = NumberInput(0, 0, 4096, 64, "number"),
    float_field: float = NumberInput(1.0, 0, 10.0, 0.01, 0.001),
    print_to_screen: str = Choice(["enabled", "disabled"]),
) -> ImageTensor:
    if print_to_screen == "enable":
        print(
            f"""Your input contains:
            string_field aka input text: {string_field}
            int_field: {int_field}
            float_field: {float_field}
        """
        )
    # do some processing on the image, in this example I just invert it
    image = 1.0 - image
    return image  # Internally this gets auto-converted to (image,) for ComfyUI.
