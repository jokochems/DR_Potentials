import os


def make_directory_if_missing(folder: str) -> None:
    """Add directories if missing (solution created querying ChatGPT)"""
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    except OSError as e:
        print(f"Failed to create directory: {e}")
