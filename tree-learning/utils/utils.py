import gc
import numpy as np

def refresh_dpt(old_dpt):
    new_dpt = {}
    for key, arr in old_dpt.items():
        if arr.size>1:  
            new_dpt[key] = arr[1:]  #remove the first element for low memory 
        else:
            new_dpt[key] = np.array([])  
    del old_dpt  
    gc.collect()  #empty old DPT's memory with garbage collector 
    return new_dpt

