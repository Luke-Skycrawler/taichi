from python.taichi.lang.diff_enhance import NlogNSim
from taichi.lang import impl
from taichi.lang import type_factory_impl as tf_impl
import numpy as np

import taichi as ti


class Quant:
    """Generator of quantized types.

    For more details, read https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf.
    """
    @staticmethod
    def int(bits, signed=False, compute=None):
        """Generates a quantized type for integers.

        Args:
            bits (int): Number of bits.
            signed (bool): Signed or unsigned.
            compute (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
        if compute is None:
            compute = impl.get_runtime().default_ip
        return tf_impl.type_factory.custom_int(bits, signed, compute)

    @staticmethod
    def fixed(frac, signed=True, range=1.0, compute=None):
        """Generates a quantized type for fixed-point real numbers.

        Args:
            frac (int): Number of bits.
            signed (bool): Signed or unsigned.
            range (float): Range of the number.
            compute (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=ti.i32)
        if signed:
            scale = range / 2**(frac - 1)
        else:
            scale = range / 2**frac
        if compute is None:
            compute = impl.get_runtime().default_fp
        return tf_impl.type_factory.custom_float(frac_type, None, compute,
                                                 scale)

    @staticmethod
    def float(exp, frac, signed=True, compute=None):
        """Generates a quantized type for floating-point real numbers.

        Args:
            exp (int): Number of exponent bits.
            frac (int): Number of fraction bits.
            signed (bool): Signed or unsigned.
            compute (DataType): Type for computation.

        Returns:
            DataType: The specified type.
        """
        # Exponent is always unsigned
        exp_type = Quant.int(bits=exp, signed=False, compute=ti.i32)
        # TODO: handle cases with frac > 32
        frac_type = Quant.int(bits=frac, signed=signed, compute=ti.i32)
        if compute is None:
            compute = impl.get_runtime().default_fp
        return tf_impl.type_factory.custom_float(significand_type=frac_type,
                                                 exponent_type=exp_type,
                                                 compute_type=compute)
    
    @staticmethod
    def auto(signed = True, compute = None):
        """Auto quantized field type
        
        Args:
            signed (bool): Signed or unsigned.
            compute (DataType): Type for computation.

        Returns:
            DataType: Only available after the pre-run
        """
        # TODO: handle cases with frac > 32
        # TODO: get a tag on the typed variables
        if compute is None:
            compute = impl.get_runtime().default_fp
        return tf_impl.type_factory.auto(signed, compute)

# @ti.data_oriented    
# class _FieldContainer:
#     """Field Container for the auto quantizer"""
#     def __init__(self, fields, snodes, prerun = True, bits = []):
#         for n,f in zip(snodes, fields):
#             n.place(f)
#         for n,f in zip(snodes, fields):
#             n.place(f.grad)
#         self.fields = fields    
        
class AutoQuant:
    """Auto-quantizer implementation
    """
    def __init__(self, n_timestep, target_func = None, loss= None, step_func=None,init = None):
        self.managed_fields = []
        self.fb = ti.FieldsBuilder()
        self.fb.finalize()
        
        self.ranges = []
        self.sim = NlogNSim(fields = self.managed_fields, n_timestep= n_timestep,target_func= target_func,loss= loss,step_func= step_func,ranges_fixed= )
    def pre_run(self):
        for _f in self.managed_fields:
            pass
        if self.init:
            self.init()
        NlogNSim.run_sim()
        
        self.fb.destroy()
    
    @staticmethod
    def coeff_fixed(a, U):
        a = np.square(a)
        a = np.sum(a)
        return a * U ** 2

    def coeff_all(self, ranges = None):
        # TODO: to get the range of each variables, 
        #   users might need to code it explicitly in peek_func
        if ranges is None:
            ranges = [1.0] * len(self.fields) 
        for sim,stash,dims,_,r in zip(self._dict, ranges):
            if r >= 0.0:    # accumulated 
                coe = self.coeff_fixed(sim.grad.to_numpy(), sim.to_numpy(), r)
                coe = max([coe, 0.25])
                self.coeffs.append(coe)
        self.coeffs = np.array(self.coeffs)
        return self.coeffs
    
    def bits_from_constrain(self, eps=1e-4, coeffs = None):
        assert len(self.coeffs) or coeffs
        if coeffs :
            self.coeffs = coeffs
        b = (self.loss * eps) ** 2
        bits = self.scipy_optimize(coeffs, b)
        return bits
    
    @staticmethod
    def scipy_optimize(coeffs, a, b):
        """solves the optimization

        Args:
            coeffs: square sum of gradients 
            a: vector, concatetation of field dimensions
            b: target error bound
        
        Returns:
            bits required
        """
        pass
# Unstable API
quant = Quant
