import os

def clean_json(img_path):
    """
    clean up json file
    :param img_path: image path
    :return: None
    """
    json_path = img_path
    os.remove(json_path)
    return None

def clean_box(img_path):
    """
    clean up box file
    :param img_path: image path
    :return: None
    """
    box_path = img_path
    os.remove(box_path)
    return None

def clean_dir(dir_path):
    root_path = dir_path
    file_names = os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, file_names[0])):
        for file_name in file_names:
            clean_dir(os.path.join(root_path, file_name))
    else:
        for file_name in file_names:
            if 'box' in file_name:
                clean_box(os.path.join(root_path, file_name))
            elif 'json' in file_name:
                clean_json(os.path.join(root_path, file_name))

if __name__ == '__main__':
    clean_dir('/Users/binah/cmu/ADE20K_2016_07_26/images/training/a/aquarium')
