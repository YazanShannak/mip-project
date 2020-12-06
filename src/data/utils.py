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
    return name.split("_")[0]


def change_id_name(original_name: str, new_id: str) -> str:
    sections = original_name.split("_")
    sections[0] = new_id
    return "_".join(sections)
