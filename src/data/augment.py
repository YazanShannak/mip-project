import random
from typing import Tuple, Any
from torchvision.transforms import functional as ft


def is_positive(name: str) -> bool:
    """
    Determines whether the sample's class is positive or negative
    Args:
        name: name of the file

    Returns:
        boolean
    """
    return True if int(name[-6]) == 1 else False


def augment_mode_1(image: Any, mask: Any) -> Tuple[Any, ...]:
    """
    Augmentation mode 1: Flips the images horizontally
    Args:
        image: input image
        mask: input mask

    Returns:
        horizontally flipped image and mask

    """
    image, mask = ft.hflip(image), ft.hflip(mask)
    return image, mask


def augment_mode_2(image: Any, mask: Any) -> Tuple[Any, ...]:
    """
    Augmentation mode 2: Consistently rotates the image and mask
    for a degree between 3-7 clockwise or counter-clockwise randomly as well
    Args:
        image: input image
        mask: input mask

    Returns:
        Rotated image and  mask
    """
    rotation_angle = random.randint(a=3, b=7)
    rotation_angle = rotation_angle if random.random() >= 0.5 else -1 * rotation_angle
    image, mask = ft.rotate(img=image, angle=rotation_angle), ft.rotate(img=mask, angle=rotation_angle)
    return image, mask
