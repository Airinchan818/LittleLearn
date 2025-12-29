"""
# LittleLearn: Touch the Big World with Little Steps 🌱

LittleLearn is an original, experimental machine learning framework — inspired by
the simplicity of Keras and the flexibility of PyTorch — but designed from scratch
with its own architecture, philosophy, and backend engine.

## 🔖 Note:

Although inspired by Keras and PyTorch, **LittleLearn is an original framework** built from the ground up.
It aims to provide a new perspective on model building — flexible, introspective, and fully customizable.

Author: 
----------
Candra Alpin Gunawan 
"""
__name__ = "littlelearn"
__author__ = "Candra Alpin Gunawan"
__version__ = "1.0.3"
__license__ = "Apache 2.0"
__realese__ = "29-December-2025"
__email__ = "hinamatsuriairin@gmail.com"
__repo__ = "https://github.com/Airinchan818/LittleLearn"
__youtube__ = "https://youtube.com/@hinamatsuriairin4596?si=KrBtOhXoVYnbBlpY"

from . import preprocessing
from .tensor import *
from .GradientReflector import GradientReflector
from .GradientReflector import Node
from . import DeepLearning


def to_tensor (x,dtype = float32,device="cpu",requires_grad=True) :
    if isinstance (x,Tensor) :
        print("warning data has being Tensor, is just rebuild Tensor in the same Tensor object")
        x = x.tensor
    return Tensor(x,dtype=dtype,device=device,requires_grad=requires_grad)

def matmul (a,b,transpose_a=False,transpose_b=False) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    if not isinstance(b,Tensor) :
        b = Tensor(b)
    return a.matmul(b,tranpose_a=transpose_a,transpose_b=transpose_b)

def sum(a,axis=None,keepdims=False) :
    if not isinstance(a,Tensor) : 
        a = Tensor(a)
    
    return a.sum(axis=axis,keepdims=keepdims)

def mean(a,axis=None,keepdims=False) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.mean(axis=axis,keepdims=keepdims)

def log (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log()

def exp(a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.exp()

def flatten (a) :
  if not isinstance(a,Tensor) :
      a = Tensor(a)
  
  return a.flatten()

def expand_dims (a,axis=0) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.expand_dims(axis=axis)

def squezze (a,axis=None) :
  if not isinstance(a,Tensor) :
      a = Tensor(a)
  
  return a.squezze(axis=axis)

def unsquezze (a,axis=None) :
    if not isinstance(a,Tensor) :
        a= Tensor(a)
    
    return a.unsquezze(axis=axis)

def sumprod (a,b) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    if not isinstance(b,Tensor) :
        b = Tensor(b)
    
    return a.sumprod(a,b)

def var(a,axis=None,keepdims=False) :
    if not isinstance(a,Tensor)  :
        a = Tensor(a)
    
    return a.var(axis=axis,keepdims=keepdims)

def std(a,axis=None,keepdims=False,epsilon=1e-5) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.std(axis=axis,keepdims=keepdims,epsilon=1e-5)

def log2 (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log2()

def log10(a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log10()

def log1p(a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log1p()

def reshape(a,new_shape: tuple) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    return a.reshape(newshape=new_shape)

def transpose(a,new_shape:tuple) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.reshape(newshape=new_shape)

def sqrt (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.sqrt()

def clip(a,min_vals :float,max_vals:float) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.clip(min_vals,max_vals)

def max(a,axis=None,keepdims=False) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.max(axis=axis,keepdims=keepdims)

def min (a,axis=None,keepdims=False) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.min(axis=axis,keepdims=keepdims)

def argmax(a,axis=None) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.argmax(axis=axis)

def abs (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.abs()

def pad(a,pad_with,mode='constant',constant_value=0) :
    if not isinstance(a,Tensor) :
        a =  Tensor(a)
    
    return a.pad(pad_with=pad_with,mode=mode,constant_value=constant_value)

def broadcast_to (a,shape :tuple) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.broadcast_to(shape)

def erf (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.erf()

def sinh (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.sinh()

def consh (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.cosh()

def tan (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.tan()

def floor (a) : 
    if not isinstance(a,Tensor) : 
        a = Tensor(a)
    
    return a.floor()

def ceil (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.ceil()

def rint (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.rint()

def reciprocal (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.reciprocal()

def sign (a) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.sign()

def identity (a) :
    if not isinstance(a,Tensor) : 
        a = Tensor(a)
    
    return a.identity()

def concat (tensor_list : list,axis=0) :
    if not isinstance(tensor_list[0],Tensor) :
        for i in range(len(tensor_list)) :
            tensor_list[i] = Tensor(tensor_list[i])
    
    return GradientReflector.concat(tensor_list,axis=axis)

def logsumexp (a,axis = None,keepdims=False) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.logsumexp(axis=axis,keepdims=keepdims)

def logaddexp (a,b) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.logaddexp(a,b)

def argmin (a,axis=None) :
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    return a.argmin(axis=axis)

def where (condition,a,b) :
   return GradientReflector.where(condition=condition,a=a,b=b)

def sin(x) :
    if not isinstance(x,Tensor) : 
        x = Tensor(x)
    return x.sin()

def cos(x) :
    if not isinstance(x,Tensor) :
        x = Tensor(x)
    
    return x.cos()