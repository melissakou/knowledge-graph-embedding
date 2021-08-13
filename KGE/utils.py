import os
import shutil

def check_path_exist_and_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
