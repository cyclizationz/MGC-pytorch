from __future__ import print_function

import os
import sys

import torch

_curr_path = os.path.abspath(__file__)
_cur_dir = os.path.dirname(_curr_path)
_torch_dir = os.path.dirname(_cur_dir)
_tool_data_dir = os.path.dirname(_torch_dir)
_deep_learning_dir = os.path.dirname(_tool_data_dir)
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir)

from .HDF5_io import *
from .MAT_io import *
from .trimesh_util import *

'''
TODO: 
1.convert lmk68 to lmk86
2.resample PCs
3.resplit LYHM datas
'''