import os
import shutil
import numpy as np
import tensorflow as tf

def check_path_exist_and_create(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)