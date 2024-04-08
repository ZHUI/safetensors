import pickle
import numpy as np
from safetensors.numpy import load_file
from safetensors.numpy import save_file

w = np.empty([256,1024,1024])

state_dict = {"weight": w}

import time
fp = "test_file.safetensors"
t1 = time.time()
save_file(state_dict,fp)
print("sf save:",  time.time()-t1)

t1 = time.time()
state = load_file(fp) 
print("sf load:", time.time()-t1)


fp = fp.replace(".safetensors", ".pickle")
t1 = time.time()
state = pickle.dump(state_dict, open(fp, "wb"))
print("pickle save:",  time.time()-t1)
 

t1 = time.time()
state = pickle.load(open(fp, "rb"))
print("pickle load:", time.time()-t1)
