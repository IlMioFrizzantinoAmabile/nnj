import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Optional, Tuple, List, Union

class AbstractJacobian:
    """Abstract class that:
    - will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    - propagate jacobian vector and jacobian matrix products, both forward and backward
    - pull back and push forward metrics
    """

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        # this function has to be implemented for every new nnj layer
        raise NotImplementedError

        
    ###############################
    ### jacobians outer product ###
    ###############################

    def jjT(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        wrt: str = "input",
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """ jacobian * jacobian.T """
        return self.jmjTp(
            x, val, None, wrt=wrt, to_diag=to_diag, diag_backprop=diag_backprop
        )

    def jTj(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        wrt: str = "input",
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        """ jacobian.T * jacobian """
        return self.jTmjp(
            x, val, None, wrt=wrt, to_diag=to_diag, diag_backprop=diag_backprop
        )


    ########################################################################################
    ### slow implementations, to be overwritten by each module for efficient computation ###
    ########################################################################################


    ######################
    ### forward passes ###
    ######################

    def _jvp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing jvp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bij,bj->bi", jacobian, vector)

    def _jmp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing jmp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def _jmjTp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing jmjTp in the stupid way!")
        if diag_backprop: #TODO
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bij,bjk,blk->bil", jacobian, matrix, jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            return torch.einsum("bij,bj,blj->bil", jacobian, matrix, jacobian)
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bij,bjk,bik->bi", jacobian, matrix, jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            return torch.einsum("bij,bj,bij->bi", jacobian, matrix, jacobian)


    #######################
    ### backward passes ###
    #######################
    
    def _vjp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing vjp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bi,bij->bj", vector, jacobian)

    def _mjp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        print(f"Ei! I ({self}) am doing mjp in the stupid way!")
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def _jTmjp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        print(f"Ei! I ({self}) am doing jTmjp in the stupid way!")
        if diag_backprop:
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None: #non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bji,bjk,bkl->bil", jacobian, matrix, jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            return torch.einsum("bij,bj,bjl->bil", jacobian, matrix, jacobian)
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bij,bjk,bki->bi", jacobian, matrix, jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            return torch.einsum("bij,bj,bji->bi", jacobian, matrix, jacobian)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Tensor,
        val2: Tensor,
        matrixes: Tuple[Tensor, Tensor, Tensor],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):  # -> Union[Tensor, Tuple]:
        print(f"Ei! I ({self}) am doing jmjTp_batch2 in the stupid way!")
        if diag_backprop:
            raise NotImplementedError
        j1 = self._jacobian(x1, val1, wrt=wrt)
        j2 = self._jacobian(x2, val2, wrt=wrt)
        if j1 is None or j2 is None: #non parametric layer
            return None
        jTmjps = []
        for j_left, matrix, j_right in ((j1, matrixes[0], j1), (j1, matrixes[1], j2), (j2, matrixes[2], j2)):
            if not from_diag and not to_diag:
                # full -> full
                jTmjps.append( torch.einsum("bji,bjk,bkl->bil", j_left, matrix, j_right) )
            elif from_diag and not to_diag:
                # diag -> full
                jTmjps.append( torch.einsum("bij,bj,bjl->bil", j_left, matrix, j_right) )
            elif not from_diag and to_diag:
                # full -> diag
                jTmjps.append( torch.einsum("bij,bjk,bki->bi", j_left, matrix, j_right) )
            elif from_diag and to_diag:
                # diag -> diag
                jTmjps.append( torch.einsum("bij,bj,bji->bi", j_left, matrix, j_right) )
        return tuple(jTmjps)