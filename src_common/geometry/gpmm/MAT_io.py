import scipy.io as sio
import numpy as np

class MAT_IO:
    def __init__(self,path_file,data_handler=None):
        if(data_handler is None):
            self.data_handler = sio.loadmat(path_file)
        else:
            self.data_handler = data_handler
    def GetMainKeys(self):
        keys = list(self.data_handler.keys())[-1]
        return keys
    def GetValue(self,key):
        return self.data_handler[key]