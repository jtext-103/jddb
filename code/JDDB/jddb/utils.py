import re
import os


def replace_pattern(directory: str, shot: int) -> str:
    """Replace the '$shot_*$' pattern to normal directory to save shots.

    Args:
        directory (str): origin input directory that may cover the pattern.
        shot (int): the shot number to save.
    Returns:
        str: replaced directory to save shots.
    """
    pattern = r'\$shot_\d+\$'
    match = re.findall(pattern, directory)
    if match:
        for each in match:
            number = 10 ** int(re.findall(r'\d+', each)[0])
            directory = directory.replace(each, '{}'.format(int(shot) // number))
    file_path = os.path.join(directory, '{}.hdf5'.format(shot))
    return file_path
