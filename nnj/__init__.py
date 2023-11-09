from nnj.abstract_diagonal_jacobian import AbstractDiagonalJacobian
from nnj.abstract_jacobian import AbstractJacobian

#########################
### parametric layers ###
from nnj.linear import Linear  # isort:skip
from nnj.conv2d import Conv2d  # isort:skip

############################
### multi-layer wrappers ###
from nnj.sequential import Sequential  # isort:skip

#############################
### non-parametric layers ###

# shape preserving, diagonal jacobian
from nnj.tanh import Tanh  # isort:skip
from nnj.relu import ReLU  # isort:skip
from nnj.sigmoid import Sigmoid  # isort:skip
from nnj.sinusoidal import Sinusoidal  # isort:skip
from nnj.softplus import Softplus  # isort:skip
from nnj.truncexp import TruncExp  # isort:skip

from nnj.reshape import Reshape  # isort:skip
from nnj.flatten import Flatten  # isort:skip

# shape preserving, non-diagonal jacobian

# non shape preserving
from nnj.upsample import Upsample  # isort:skip     # missing forward passes
from nnj.maxpool2d import MaxPool2d  # isort:skip   # missing forward passes

#############################
### utils                 ###
from nnj.utils import convert_to_nnj, invert_block_diagonal  # isort:skip
from torch.nn.utils import vector_to_parameters, parameters_to_vector  # isort:skip
