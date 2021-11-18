# from mpm_exp.contrast_oop import scipy_optimize
import numpy as np

import taichi as ti

@ti.kernel
def _clear_field(f: ti.template(), odd: ti.i32, dims: ti.template()):
    for p in ti.grouped(ti.ndrange(*dims)):
        f.grad[odd, p].fill(0.0)

@ti.kernel
def _clear_field_scala(f: ti.template(), odd: ti.i32, dims: ti.template()):
    for p in ti.grouped(ti.ndrange(*dims)):
        f.grad[odd, p] = 0.0

def _cnt_ending_zero(s: ti.i32) -> ti.i32:
    cnt = 0
    s = s+1
    while s % 2 == 0:
        cnt += 1
        s /= 2
    return cnt

@ti.kernel
def _copy_field(src: ti.template(), dst: ti.template(), odd: ti.i32, c: ti.i32, dims: ti.template()):
    for p in ti.grouped(ti.ndrange(*dims)):
        dst[c,p] = src[odd,p]

@ti.data_oriented
class NlogNSim:
    """USUAGE:
      0. implement a simulation by with double buffer on 
          any differentiable variables
    
      1. define the main simulation function 
          with a single parameter 's' being the step index
    
      2. specify all required arguments in constructor
    
      3. after setting up the initial states, 
          'run_sim' funciton takes care of the rest
    
    methods starting with _ are utility functions
    """
    
    def __init__(self, *, n_timestep = None, fields = None, target_func = None, loss = None, step_func = None, peek_func = None, ranges_fixed = None):
        assert n_timestep and fields and step_func
        assert not ((loss is not None) ^ (target_func is not None))
        self.n_timestep = n_timestep
        self.logn_timestep = int(np.ceil(np.log2(n_timestep))+1)
        self.loss = loss
        self.fields = fields
        self.single_step, self.target_func = step_func, target_func
        self.peek_func = peek_func
        self._stashed_fields = []
        self._dims = []
        self._is_scala = []
        self.coeffs = []
        for f in self.fields:
            assert f.shape[0] == 2
            _, *dims = f.shape
            _type = str(type(f)).split('.')[-1][:-2]
            print(_type)
            if _type == 'ScalarField':
                self._stashed_fields.append(ti.field(dtype = f.dtype, shape = (self.logn_timestep, *dims)))
                self._is_scala.append(True)
            elif _type == 'MatrixField':
                self._stashed_fields.append(ti.Matrix.field(f.n, f.m, dtype = f.dtype, shape = (self.logn_timestep, *dims)))
                self._is_scala.append(False)
            self._dims.append(tuple(dims))
            
        self._dict = [[i,j,k,l] for i,j,k,l in zip(self.fields, self._stashed_fields, self._dims, self._is_scala)]
        print(len(self._dict), len(self.fields), len(self._stashed_fields), len(self._is_scala),len(self._dims))
        self._c = self.logn_timestep-1

    @ti.ad.grad_replaced
    def _single_step(self,s):
        """define everything to watch here        
        """
        if self._c and (s+1) % (1 << (self._c-1)) == 0:
            self._c -= 1
            self._checkpoint(self._c, 1-s % 2)

    def _checkpoint(self,c: ti.i32, odd: ti.i32):
        """saves x[odd] to checkpoint c
        """
        for sim,stash,dims,_ in self._dict:
            _copy_field(sim, stash, odd, c, dims)

    def _load_checkpoint(self, w: ti.i32):
        """loads from checkpoints w and puts in x[0]
        """
        for sim,stash,dims,_ in self._dict:
            _copy_field(stash, sim, w, 0, dims)

    def _rerun_from(self, N, w):
        """N, w: total steps, current step
        """
        self._load_checkpoint(min([w+1, self.logn_timestep-1]))
        self._checkpoint(min(w, self.logn_timestep-1), 0)

        c = w
        for s in range(N-N % (1 << (w+1)), N):
            self.single_step(s)
            if c and (s+1) % (1 << (c-1)) == 0:
                c -= 1
                self._checkpoint(c, 1-s % 2)
        assert c == 0
    
    def _clear_grad(self, odd: ti.i32):
        """clear *.grad[odd]
        """
        for f,_,dims,is_scala in self._dict:
            if is_scala:
                _clear_field_scala(f, odd, dims)
            else :
                _clear_field(f, odd, dims)
    
    @ti.ad.grad_replaced
    def _peek(self,s):
        pass

    @ti.ad.grad_for(_peek)
    def peek(self, s):
        """add watches over the grads by overriding this funtion 
        """
        if not self.peek_func:
            return

        self.peek_func(s)

    @ti.ad.grad_for(_single_step)
    def _single_step_grad(self,s):
        w = _cnt_ending_zero(s)
        self._rerun_from(s,w)
        self.single_step(s)
        self._clear_grad(s % 2)

    def run_sim(self):
        self._checkpoint(self.logn_timestep-1,0)
        if self.loss and self.target_func:
            with ti.Tape(loss = self.loss):
                for s in range(self.n_timestep):
                    if self.peek_func:
                        self._peek(s) # observation fuction for grad after step s
                    self.single_step(s)
                    self._single_step(s)
                self.target_func()
        
__all__ = [NlogNSim]