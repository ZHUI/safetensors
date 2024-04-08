# cython: profile=True
import pickle
import numpy as np
from safetensors.numpy import load_file
from safetensors.numpy import save_file
import cython

@cython.cfunc
@cython.inline
def my_often_called_function():
    fp = "test_file.safetensors"
    import time
    t1 = time.time()
    state = load_file(fp) 
    print("sf load:", time.time()-t1)

my_often_called_function()
