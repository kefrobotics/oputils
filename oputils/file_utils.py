import os
from natsort import natsorted

from loguru import logger


def create_dir(directory: str, warning: str = None):
    """ Bag utility function for creating a directory

    Args:
        directory (str): Path to create directory
        warning (str, optional): optional warning message. Defaults to None.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if warning is None:
            logger.warning(f"Directory already exists: {directory}")
        elif warning == "":
            ...
        else:
            print(warning)


def rename_images(directory: str):
    """ Rename images in directory
    """
    # Get a list of image files in the directory, assuming they are PNGs
    image_files = natsorted([f for f in os.listdir(directory) if f.endswith('.png')])

    # Iterate through the list and rename the files
    for index, old_name in enumerate(image_files, start=1):
        # New file name: index.png
        new_name = f"{index}.png"

        # Full old and new file paths
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
