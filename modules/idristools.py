""" Set of tools I find useful in all my programs

This module contains functions I have found useful in other programs,
previously I was copy and pasting them in to each program separately.

    Methods:
        fancy_print: Makes string output to the console look nicer

        get_json: Finds json file and returns it as dict

        save_to_file: Saves data to file in specified path

        debug_stats: Prints out a json file/dictionary nicely

        unread_files: Scans a directory for files and determines how many of
        them have been read by the program previously

        append_to_file: Appends a string to the end of a file
"""

__author__ = "Idris Hayward"
__copyright__ = "2021, The University of Surrey & National Physical Laboratory"
__credits__ = ["Idris Hayward"]
__license__ = "GNU General Public License v3.0"
__version__ = "1.0"
__maintainer__ = "Idris Hayward"
__email__ = "j.d.hayward@surrey.ac.uk"
__status__ = "Stable Release"

import os
import json

def fancy_print(
    str_to_print,
    length=120,
    form="NORM",
    char="\U0001F533",
    end="\n",
    flush=False,
):
    """Makes strings output to the console look nicer

    This function is used to make the console output of python
    scripts look nicer. This function is used in a range of
    modules to save time in formatting console output.

        Keyword arguments:
            str_to_print (str): The string to be formatted and printed

            length (int): Total length of formatted string

            form (str): String indicating how 'str_to_print' will be
            formatted. The options are:
                'TITLE': Centers 'str_to_print' and puts one char at
                         each end
                'NORM': Left justifies 'str_to_print' and puts one char
                        at each end (Norm doesn't have to be specified,
                        any option not listed here has the same result)
                'LINE': Prints a line of 'char's 'length' long

            char (str): The character to print.

        Variables:
            length_slope (float): Slope used to adjust length of line.
            Used if an emoji is used for 'char' as they take up twice
            as much space. If one is detected, the length is adjusted.

            length_offset (int): Offset used to adjust length of line.
            Used if an emoji is used for 'char' as they take up twice
            as much space. If one is detected, the length is adjusted.

        Returns:
            Nothing, prints a 'form' formatted 'str_to_print' of
            length 'length'
    """
    length_adjust = 1
    length_offset = 0
    if len(char) > 1:
        char = char[0]
    if len(char.encode("utf-8")) > 1:
        length_adjust = 0.5
        length_offset = 1
    if form == "TITLE":
        print(
            f"{char} {str_to_print.center(length - 4, ' ')} {char}",
            end=end,
            flush=flush,
        )
    elif form == "LINE":
        print(
            char * int(((length) * length_adjust) + length_offset),
            end=end,
            flush=flush,
        )
    else:
        print(
            f"{char} {str_to_print.ljust(length - 4, ' ')} {char}",
            end=end,
            flush=flush,
        )


def get_json(path_to_json):
    """Finds json file and returns it as dict

    Creates blank file with required keys at path if json file is not
    present

        Keyword Arguments:
            path_to_json (str): Path to the json file, including
            filename and .json extension

        Returns:
            Dict representing contents of json file

        Raises:
            FileNotFoundError if file is not present, blank file created

            ValueError if file can not be parsed
    """

    try:
        with open(path_to_json, "r") as jsonFile:
            try:
                return json.load(jsonFile)
            except json.decoder.JSONDecodeError:
                raise ValueError(
                    f"{path_to_json} is not in the proper"
                    f"format. If you're having issues, consider"
                    f"using the template from the Github repo or "
                    f" use the format seen in README.md"
                )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{path_to_json} could not be found, use the "
            f"template from the Github repo or use the "
            f"format seen in README.md"
        )


def save_to_file(write_data, path_to_file, filename):
    """Saves data to file in specified path

    This format agnostic function saves a string of data to a specified file.
    File format and path to file are determined by keyword arguments,
    this just writes data (preferrably string format)

    Keyword Arguments:
        write_data (str): Data to be written to file, preferably in string
        format

        path_to_file (str): The path that the file will be created in,
        if it's not present it will be created

        filename (str): Name of file to write data to, please include file
        format e.g "filename.csv"

    Returns:
        None
    """
    while not os.path.isdir(path_to_file):
        os.makedirs(path_to_file, exist_ok=True)
    with open(f"{path_to_file}/{filename}", "w") as newfile:
        newfile.write(write_data)


def debug_stats(stats, line_length=120, level=1, max_level=3):
    """ Prints out a json file/dictionary nicely

    Used to print out config files in an easily readable way, can also be used
    for other dictionary type formats. Will print nested dictionaries up to a 
    depth of max_level.

    Keyword arguments:
        stats (dict): Dictionary to print to console

        line_length (int): Length of the console

        level(int): How far the dictionary is nested, the user should never use
        this keyword when calling this function.

        max_level (int): How deep in to a nested dictionary the function will print

    Returns:
        None
    """
    base_string = f"{'    '*level}- "
    for key, item in stats.items():
        if type(item) == dict and level <= max_level:
            fancy_print(f"{base_string}{key}:", length=line_length)
            debug_stats(item, level=level+1)
        elif type(item) == dict:
            fancy_print(f"{base_string}{key}: {...}", length=line_length)
        else:
            stat_string = f"{base_string}{key}: {item}"
            if len(stat_string) > line_length - 10:
                stat_string = f"{stat_string[:line_length-15]}..."
            fancy_print(stat_string, length=line_length)


def unread_files(path, read_list, return_stats=False):
    """ Scans a directory for files and determines how many of them have been
    read by the program previously

    Scans a directory for a list of files and then removes them from the list
    if the program has read them previously. This function requires that files
    read by the master program are added to a list after they have been
    processed.

    Keyword arguments:
        path (str): Path to directory containing files to be read

        read_list (str): Path to file containing all previously read files. The
        path must include the file extension.

        return_stats (boolean): If true, a dictionary containing the list of
        unread files, the number of files in total and the number of read files
        is returned. Otherwise, the list of unread files is returned on its
        own. Defaults to False.

    Returns:
        A list of unread files if return_stats is false
        A dictionary containing:
            "Unread File List": A list of unread files
            "Total Files": The number of files in the directory
            "Read Files": The number of read files
        if return_stats is true
    """
    file_list = os.listdir(path)
    file_list.sort()
    try:
        with open(read_list, "r") as read_files_txt:
            read_files = read_files_txt.readlines()
            read_files = [line[:-1] for line in read_files]
    except FileNotFoundError:
        with open(read_list, "w") as read_files_txt:
                read_files = list()
    if len(read_files) == 0:
        unread_files = file_list
    else:
        unread_files = [file for file in file_list if file not in read_files]
    if return_stats:
        return {
            "Unread File List": unread_files,
            "Total Files": len(file_list),
            "Read Files": len(read_files)
            }
    else:
        return unread_files


def append_to_file(line, file):
    """ Appends a line to a file

    Keyword arguments:
        line (str): What to append to the file

        file (str): Path to the file to append to

    Returns:
        None
    """
    with open(file, "a") as read_files_txt:
        read_files_txt.write(f"{line}\n")
