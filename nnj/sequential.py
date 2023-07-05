import torch
from torch import nn, Tensor
from nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Sequential(AbstractJacobian, nn.Sequential):

    def __init__(self, *args, add_hooks: bool = False):
        super().__init__(*args)
        self._modules_list = list(self._modules.values())

        self._n_params = 0
        for k in range(len(self._modules)):
            self._n_params += self._modules_list[k]._n_params

        self.add_hooks = add_hooks
        if self.add_hooks:
            self.feature_maps = []
            self.handles = []
            # def fw_hook(module, input, output):
            #    self.feature_maps.append(output.detach())
            for k in range(len(self._modules)):
                # self.handles.append(self._modules_list[k].register_forward_hook(fw_hook))
                self.handles.append(
                    self._modules_list[k].register_forward_hook(
                        lambda m, i, o: self.feature_maps.append(o.detach())
                    )
                )

    def forward(
        self, x: Tensor, jacobian: Union[Tensor, bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.add_hooks:
            self.feature_maps = [x]
        #if not (jacobian is False): # if jacobian is true or is instance of tensor
        #    j = identity(x) if (not isinstance(jacobian, Tensor) and jacobian) else jacobian
        if jacobian is True: 
            j = None
        elif isinstance(jacobian, Tensor):
            j = jacobian
        for module in self._modules.values():
            val = module(x)
            if not (jacobian is False):
                j = module._jmp(x, val, j)
            x = val
        if not (jacobian is False):
            return x, j
        return x
    
    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
    ) -> Tensor:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            xs = x.shape
            identity_x = (
                torch.eye(xs[1:].numel(), xs[1:].numel(), dtype=x.dtype, device=x.device)
                .repeat(xs[0], 1, 1)
            )
            val, jmp = self.forward(x, jacobian = identity_x)
            return jmp
        elif wrt == "weight":
            if val is None:
                val = self.forward(x)
            vs = val.shape
            identity_val = (
                torch.eye(vs[1:].numel(), vs[1:].numel(), dtype=val.dtype, device=val.device)
                .repeat(vs[0], 1, 1)
            )
            return self._mjp(x, val, identity_val, wrt="weight")


    ######################
    ### forward passes ###
    ######################

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            for module in self._modules.values():
                val = module(x)
                vector = module._jvp(x, val, vector, wrt=wrt)
                x = val
            return vector
        elif wrt == "weight":
            p = 0
            jvp = None
            for module in self._modules.values():
                val = module(x)
                jvp = module._jvp(x, val, jvp, wrt="input") if jvp is not None else None
                jvp_from_layer = module._jvp(x, val, vector[:, p: p+module._n_params], wrt="weight")
                if jvp_from_layer is not None:
                    if jvp is None:
                        jvp = jvp_from_layer
                    else:
                        jvp += jvp_from_layer
                p += module._n_params
                x = val
            assert p == self._n_params
            return jvp
    

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            for module in self._modules.values():
                val = module(x)
                matrix = module._jmp(x, val, matrix, wrt=wrt)
                x = val
            return matrix
        elif wrt == "weight":
            p = 0
            jmp = None
            for module in self._modules.values():
                val = module(x)
                jmp = module._jmp(x, val, jmp, wrt="input") if jmp is not None else None
                jmp_from_layer = module._jmp(x, val, matrix[:, p: p+module._n_params, :], wrt="weight")
                if jmp_from_layer is not None:
                    if jmp is None:
                        jmp = jmp_from_layer
                    else:
                        jmp += jmp_from_layer
                p += module._n_params
                x = val
            assert p == self._n_params
            return jmp
        
    def _jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian matrix jacobian.T product
        """
        # forward pass
        if val is None:
            val = self.forward(x)
        if wrt == "input":
            if matrix is None:
                matrix = torch.ones_like(x)
                from_diag = True
            # forward pass again
            for k in range(len(self._modules_list)):
                # propagate through the input
                matrix = self._modules_list[k]._jmjTp(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    matrix,
                    wrt="input",
                    from_diag=from_diag if k == 0 else diag_backprop,
                    to_diag=to_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    diag_backprop=diag_backprop,
                )
            return matrix
        elif wrt == "weight":
            if matrix is None:
                matrix = torch.ones((x.shape[0], self._n_params), dtype=x.dtype, device=x.device)
                from_diag = True
            # forward pass again
            assert (from_diag and isinstance(matrix, Tensor)) or (not from_diag and isinstance(matrix, list))
            jmjTp = None
            p = 0
            for k in range(len(self._modules_list)):
                # propagate through the input
                if jmjTp is not None:
                    jmjTp = self._modules_list[k]._jmjTp(
                        self.feature_maps[k],
                        self.feature_maps[k + 1],
                        jmjTp,
                        wrt="input",
                        from_diag=from_diag if k == 0 else diag_backprop,
                        to_diag=to_diag if k == len(self._modules_list) - 1 else diag_backprop,
                        diag_backprop=diag_backprop,
                    )
                # propagate through the weight
                jmjTp_from_layer = self._modules_list[k]._jmjTp(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    matrix[k] if not from_diag else matrix[:, p: p+self._modules_list[k]._n_params],
                    wrt="weight",
                    from_diag=from_diag if k == 0 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if jmjTp_from_layer is not None:
                    if jmjTp is None:
                        jmjTp = jmjTp_from_layer
                    else:
                        jmjTp += jmjTp_from_layer
                p += self._modules_list[k]._n_params
            return jmjTp
        

    #######################
    ### backward passes ###
    #######################

    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        # forward pass for computing hook values
        if val is None:
            val = self.forward(x)

        # backward pass
        vs = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # backpropagate through the weight
            if wrt == "weight":
                v_k = self._modules_list[k]._vjp(
                    self.feature_maps[k], self.feature_maps[k + 1], vector, wrt="weight"
                )
                if v_k is not None:
                    vs = v_k + vs if isinstance(v_k, list) else [v_k] + vs 
                if k == 0:
                    break
            # backpropagate through the input
            vector = self._modules_list[k]._vjp(
                self.feature_maps[k], self.feature_maps[k + 1], vector, wrt="input"
            )
        if wrt == "weight":
            return torch.cat(vs, dim=1)
        elif wrt == "input":
            return vector

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        # forward pass
        if val is None:
            val = self.forward(x)
        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # backpropagate through the weight
            if wrt == "weight":
                m_k = self._modules_list[k]._mjp(
                    self.feature_maps[k], self.feature_maps[k + 1], matrix, wrt="weight"
                )
                if m_k is not None:
                    ms = m_k + ms if isinstance(m_k, list) else [m_k] + ms
                if k == 0:
                    break
            # backpropagate through the input
            matrix = self._modules_list[k]._mjp(
                self.feature_maps[k], self.feature_maps[k + 1], matrix, wrt="input"
            )
        if wrt == "weight":
            return torch.cat(ms, dim=2)
        elif wrt == "input":
            return matrix


    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product
        """
        # forward pass
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones((val.shape[0], val.shape[1:].numel()))
            from_diag = True
        # backward pass
        ms = []
        for k in range(len(self._modules_list) - 1, -1, -1):
            # backpropagate through the weight
            if wrt == "weight":
                m_k = self._modules_list[k]._jTmjp(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    matrix,
                    wrt="weight",
                    from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if m_k is not None:
                    ms = m_k + ms if isinstance(m_k, list) else [m_k] + ms
                if k == 0:
                    break
            # backpropagate through the input
            matrix = self._modules_list[k]._jTmjp(
                self.feature_maps[k],
                self.feature_maps[k + 1],
                matrix,
                wrt="input",
                from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                to_diag=to_diag if k == 0 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrix
        elif wrt == "weight":
            if len(ms) == 0: # case of a Sequential with no parametric layers inside
                return None  
            if to_diag:      # diagonal case returns the tensor of the diagonal
                return torch.cat(ms, dim=1)
            else:            # non diagonal case returns a list of tensor, one for each layer
                return ms

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ):  # -> Union[Tensor, Tuple]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape

        # forward passes
        self.forward(x1)
        feature_maps_1 = self.feature_maps
        self.forward(x2)
        feature_maps_2 = self.feature_maps

        if matrixes is None:
            matrixes = tuple(torch.ones_like(self.feature_maps[-1]) for _ in range(3))
            from_diag = True

        # backward pass
        ms = tuple(([], [], []))
        for k in range(len(self._modules_list) - 1, -1, -1):
            # print('layer:',self._modules_list[k])
            if wrt == "weight":
                m_k = self._modules_list[k]._jTmjp_batch2(
                    feature_maps_1[k],
                    feature_maps_2[k],
                    feature_maps_1[k + 1],
                    feature_maps_2[k + 1],
                    matrixes,
                    wrt="weight",
                    from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                    to_diag=to_diag,
                    diag_backprop=diag_backprop,
                )
                if m_k is not None:
                    if all(isinstance(m_k[i], list) for i in range(3)):
                        ms = tuple(m_k[i] + ms[i] for i in range(3) )
                    else:
                        #print(m_k[0].shape)
                        ms = tuple([m_k[i]] + ms[i] for i in range(3) )
                if k == 0:
                    break
            matrixes = self._modules_list[k]._jTmjp_batch2(
                feature_maps_1[k],
                feature_maps_2[k],
                feature_maps_1[k + 1],
                feature_maps_2[k + 1],
                matrixes,
                wrt="input",
                from_diag=from_diag if k == len(self._modules_list) - 1 else diag_backprop,
                to_diag=to_diag if k == 0 else diag_backprop,
                diag_backprop=diag_backprop,
            )
        if wrt == "input":
            return matrixes
        elif wrt == "weight":
            if all(len(ms[i]) for i in range(3)) == 0:
                return None  # case of a Sequential with no parametric layers
            if to_diag:
                return tuple( torch.cat(m, dim=1) for m in ms)
            else:
                return ms