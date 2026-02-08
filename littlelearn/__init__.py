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
__version__ = "1.1.4"
__license__ = "Apache 2.0"
__realese__ = "8-February-2026"
__email__ = "hinamatsuriairin@gmail.com"
__repo__ = "https://github.com/Airinchan818/LittleLearn"

from . import preprocessing
from .tensor import *
from .GradientReflector import GradientReflector
from .GradientReflector import Node
from . import DeepLearning


def to_tensor (x,dtype = float32,device="cpu",requires_grad=True) :
    """
    Convert input data to a Tensor object.
    Paramters :
    -----------
    x : array-like
        Input data to be converted to Tensor.
    dtype : data type, default=float32
        Desired data type of the Tensor.
    device : str, default="cpu"
        Device where the Tensor will be stored ("cpu" or "gpu").
    requires_grad : bool, default=True
        If True, gradients will be computed for this Tensor during backpropagation.
    """
    if isinstance (x,Tensor) :
        print("warning data has being Tensor, is just rebuild Tensor in the same Tensor object")
        x = x.tensor
    return Tensor(x,dtype=dtype,device=device,requires_grad=requires_grad)

def matmul (a,b,transpose_a=False,transpose_b=False) :
    """
    Perform matrix multiplication between two Tensors.
    Parameters :
    -----------
    a : Tensor or array-like
        First input Tensor.
    b : Tensor or array-like
        Second input Tensor.
    transpose_a : bool, default=False
        If True, transpose the first Tensor before multiplication.
    transpose_b : bool, default=False
        If True, transpose the second Tensor before multiplication.
    Returns :
    --------
    Tensor
        Result of the matrix multiplication.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    if not isinstance(b,Tensor) :
        b = Tensor(b)
    return a.matmul(b,tranpose_a=transpose_a,transpose_b=transpose_b)

def sum(a,axis=None,keepdims=False) :
    """
    Compute the sum of all elements in the Tensor along a specified axis.
    Parameters :
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the sum. Default is None, which sums all elements.
    keepdims : bool, default=False
        If True, retains reduced dimensions with length 1.
    Returns :
    --------
    Tensor
        Sum of the elements along the specified axis.

    """
    if not isinstance(a,Tensor) : 
        a = Tensor(a)
    
    return a.sum(axis=axis,keepdims=keepdims)

def mean(a,axis=None,keepdims=False) :
    """
    Compute the mean of all elements in the Tensor along a specified axis.
    
    Parameters:
    ----------
    a : Tensor or array-like
        Input Tensor.
    
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the mean. Default is None, which computes the mean of all elements.
    keepdims : bool, default=False
        If True, retains reduced dimensions with length 1.
    Returns:
    -------
    Tensor
        Mean of the elements along the specified axis.
    """
    if not isinstance(a,Tensor) :
        
        a = Tensor(a)
    
    return a.mean(axis=axis,keepdims=keepdims)

def log (a) :
    """
    Compute the natural logarithm of each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Tensor containing the natural logarithm of each element.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log()

def exp(a) :
    """
    Compute the exponential of each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Tensor containing the exponential of each element.

    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.exp()

def flatten (a) :
  """
    Flatten the input Tensor into a 1D array.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Flattened 1D Tensor.
  """
  if not isinstance(a,Tensor) :
      a = Tensor(a)
  
  return a.flatten()

def expand_dims (a,axis=0) :
    """
    Expand the dimensions of the input Tensor by adding a new axis at the specified position.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int, default=0
        Position where the new axis is to be inserted.
    Returns:
    --------
    Tensor
        Tensor with expanded dimensions.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.expand_dims(axis=axis)

def squeeze (a,axis=None) :
    """
    Remove single-dimensional entries from the shape of the Tensor.
    Parameters:
    -----------


    """
    if not isinstance(a,Tensor)  :
      a = Tensor(a)
    return a.squezze(axis=axis)


def unsqueeze (a,axis=None) :
    """
    add single-dimensional entries to the shape of the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int, default=None
        Position where the new axis is to be inserted.
    Returns:
    --------
    Tensor
        Tensor with expanded dimensions.

    """
    if not isinstance(a,Tensor) :
        a= Tensor(a)
    
    return a.unsquezze(axis=axis)

def sumprod (a,b) :
    """
    Compute the sum of the element-wise product of two Tensors.
    Parameters:
    -----------
    a : Tensor or array-like
        First input Tensor.
    b : Tensor or array-like
        Second input Tensor.
    Returns:
    --------
    Tensor
        Sum of the element-wise product of the two Tensors.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    if not isinstance(b,Tensor) :
        b = Tensor(b)
    
    return a.sumprod(a,b)

def var(a,axis=None,keepdims=False) :
    """
    Compute the variance of the elements in the Tensor along a specified axis.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the variance. Default is None, which computes the variance
        of all elements.
    keepdims : bool, default=False
        If True, retains reduced dimensions with length 1.
    Returns:
    --------
    Tensor
        Variance of the elements along the specified axis.
    """
    if not isinstance(a,Tensor)  :
        a = Tensor(a)
    
    return a.var(axis=axis,keepdims=keepdims)

def std(a,axis=None,keepdims=False,epsilon=1e-5) :
    """
    Compute the standard deviation of the elements in the Tensor along a specified axis.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the standard deviation. Default is None, which computes the
        standard deviation of all elements.
    keepdims : bool, default=False
        If True, retains reduced dimensions with length 1.
    epsilon : float, default=1e-5
        Small value added to variance to prevent division by zero.
    Returns:
    --------
    Tensor
        Standard deviation of the elements along the specified axis.

    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.std(axis=axis,keepdims=keepdims,epsilon=1e-5)

def log2 (a) :
    """
    Compute the base-2 logarithm of each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Tensor containing the base-2 logarithm of each element.

    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log2()

def log10(a) :
    """
    Compute the base-10 logarithm of each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Tensor containing the base-10 logarithm of each element.

    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log10()

def log1p(a) :
    """
    Compute the natural logarithm of one plus each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Tensor containing the natural logarithm of one plus each element.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.log1p()

def reshape(a,new_shape: tuple) :
    """
    Reshape the input Tensor to a new shape.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    new_shape : tuple
        Desired shape for the output Tensor.
    Returns:
    --------
    Tensor
        Reshaped Tensor.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    return a.reshape(newshape=new_shape)

def transpose(a,new_shape:tuple) :
    """ 
    Transpose the input Tensor according to the specified new shape.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    new_shape : tuple
        Desired shape for the transposed Tensor.
    Returns:
    --------
    Tensor
        Transposed Tensor.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.transpose(shape=new_shape)

def sqrt (a) :
    """
    Compute the square root of each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------
    Tensor
        Tensor containing the square root of each element.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.sqrt()

def clip(a,min_vals :float,max_vals:float) :
    """
    Clip (limit) the values in the Tensor to a specified minimum and maximum.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    min_vals : float
        Minimum value to clip to.
    max_vals : float
        Maximum value to clip to.
    Returns:
    --------
    Tensor
        Tensor with values clipped to the specified range.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.clip(min_vals,max_vals)

def max(a,axis=None,keepdims=False) :
    """
    Compute the maximum value along a specified axis of the input Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int, optional
        Axis along which to compute the maximum. If None, computes the maximum over all elements.
    keepdims : bool, default=False
        If True, retains reduced dimensions with length 1.
    Returns:
    --------
    Tensor
        Maximum value along the specified axis.

    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.max(axis=axis,keepdims=keepdims)

def min (a,axis=None,keepdims=False) :
    """
    Compute the minimum value along a specified axis of the input Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int, optional
        Axis along which to compute the minimum. If None, computes the minimum over all elements.
    keepdims : bool, default=False
        If True, retains reduced dimensions with length 1.
    Returns:
    --------
    Tensor
        Minimum value along the specified axis.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.min(axis=axis,keepdims=keepdims)

def argmax(a,axis=None) :
    """
    Compute the indices of the maximum values along a specified axis of the input Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    axis : int, optional
        Axis along which to compute the indices of the maximum values. If None, computes over the flattened array.
    Returns:
    --------
    Tensor
        Indices of the maximum values along the specified axis.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.argmax(axis=axis)

def abs (a) :
    """
    Compute the absolute value of each element in the Tensor.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    Returns:
    --------    
    Tensor
        Tensor containing the absolute values of each element.
    """
    if not isinstance(a,Tensor) :
        a = Tensor(a)
    
    return a.abs()

def pad(a,pad_with,mode='constant',constant_value=0) :
    """
    Pad the input Tensor with specified values along its edges.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    pad_with : tuple
        Tuple specifying the number of values to pad on each axis.
    mode : str, default='constant'
        Padding mode. Options include 'constant', 'edge', 'reflect', etc.
    constant_value : float, default=0
        Value to use for constant padding (if mode is 'constant').
    Returns:
    --------
    Tensor
        Padded Tensor.
    """
    if not isinstance(a,Tensor) :
        a =  Tensor(a)
    
    return a.pad(pad_with=pad_with,mode=mode,constant_value=constant_value)

def broadcast_to (a,shape :tuple) :
    """
    Broadcast the input Tensor to a new shape.
    Parameters:
    -----------
    a : Tensor or array-like
        Input Tensor.
    shape : tuple
        Desired shape for the broadcasted Tensor.
    Returns:
    --------
    Tensor
        Broadcasted Tensor.
    """
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
