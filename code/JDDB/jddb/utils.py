import re
import os


def replace_pattern(directory: str, shot: int, include_filename: bool = False) -> str:
    """Replace the '$shot_*$' pattern to normal directory to save shots.

    Args:
        directory (str): origin input directory that may cover the pattern.
        shot (int): the shot number to save.
        include_filename: use file name as the base_path
    Returns:
        str: replaced directory to save shots.
    """
    pattern = r'\$shot_\d+\$'
    match = re.findall(pattern, directory)
    if match:
        for each in match:
            number = 10 ** int(re.findall(r'\d+', each)[0])
            directory = directory.replace(each, '{}'.format(int(shot) // number))
    if include_filename:
        directory = directory.replace(r'$shot$', '{}'.format(shot))
        file_path = directory
    else:
        file_path = os.path.join(directory, '{}.hdf5'.format(shot))
    return file_path
