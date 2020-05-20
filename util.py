import pickle
import os

TEMP_DIR = 'tmp'


def save_pickle(obj, filename):
    if not os.path.isdir(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    path = os.path.join(TEMP_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    if not os.path.isdir(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    path = os.path.join(TEMP_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


def is_exist(filename):
    path = os.path.join(TEMP_DIR, filename)
    return os.path.isfile(path)

