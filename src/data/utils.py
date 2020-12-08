def is_positive(name: str) -> bool:
    """
    Determines whether the sample's class is positive or negative
    Args:
        name: name of the file

    Returns:
        boolean
    """
    return True if int(name[-6]) == 1 else False


def get_id(name: str) -> str:
    """
    Parse file (image) id from name
    Args:
        name: File (image) name

    Returns:
        string id

    """
    return name.split("_")[0]


def change_id_name(original_name: str, new_id: str) -> str:
    """
    Changes id for a given file (image) name
    Args:
        original_name: Original file (image) string name
        new_id: New string id

    Returns:
        New string name
    """
    sections = original_name.split("_")
    sections[0] = new_id
    return "_".join(sections)


def is_train(name: str) -> bool:
    """
    Determines whether this image is part of the training set or not
    Args:
        name: File (image) string name

    Returns:
        True if is part of training set, otherwise False

    """

    return name.split("_")[1] == "train"
