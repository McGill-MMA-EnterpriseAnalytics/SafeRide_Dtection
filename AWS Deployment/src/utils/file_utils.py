import os

def ensure_folder_exists(folder_path):
    """
    Creates a folder if it doesn't already exist.

    Args:
        folder_path (str): Path to the folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
