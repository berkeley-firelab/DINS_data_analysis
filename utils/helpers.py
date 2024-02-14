import pickle


def save_python_objects(pyobject=None, save_path=None):
    """
    Saves sklearn or any other python object as pickle file

    :param pyobject: object to be saved
    :param save_path: path to save file in string format
    :returns: NA
    :rtype: NA

    """
    with open(save_path, "wb") as f:
        pickle.dump(pyobject, f)