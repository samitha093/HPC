import os
import imageio
import re


def sort_strings_with_suffix(strings):
    """
    The function `sort_strings_with_suffix` sorts a list of strings based on a numerical suffix
    extracted from each string using regular expressions.
    
    Args:
      strings: A list of strings that need to be sorted. Each string may or may not have a numeric
    suffix at the end.
    
    Returns:
      a sorted list of strings, where the sorting is based on the suffix of each string.
    """
    def extract_suffix(string):
        # Extract the suffix using regular expression
        suffix = re.findall(r'\d+', string)
        
        if suffix:
            return int(suffix[0])
        else:
            return string

    return sorted(strings, key=lambda s: extract_suffix(s))


def generate_gif(path, file_prefix, output_path, output_file, duration):
    """
    The function generates a GIF by reading a series of images from a specified path, sorting them based
    on a file prefix, and saving the resulting GIF to an output path with a specified duration.
    
    Args:
      path: The path parameter is the directory where the images are located.
      file_prefix: The file prefix is a string that is used to filter the files in the given path. Only
    the files that have the file prefix in their names will be included in the GIF generation process.
      output_path: The output_path parameter is the path where the generated GIF file will be saved.
      output_file: The output_file parameter is the name of the GIF file that will be generated. It
    should include the file extension, such as ".gif".
      duration: The duration parameter specifies the time duration (in seconds) for each frame in the
    generated GIF.
    """
    # Create the images folder if it doesn't exist
    output_path = "images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Create the frames
    frames = []
    list_files = os.listdir(path)
    list_files_sorted = sort_strings_with_suffix(list_files)
    for file in list_files_sorted:
        if file_prefix in file:
            image = imageio.v2.imread(os.path.join(path,file)) 
            frames.append(image)
    imageio.mimsave(os.path.join(output_path, output_file),
                frames,          
                duration = duration)