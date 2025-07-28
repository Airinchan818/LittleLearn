"""
# LittleLearn: Touch the Big World with Little Steps 🌱

LittleLearn is an original, experimental machine learning framework — inspired by
the simplicity of Keras and the flexibility of PyTorch — but designed from scratch
with its own architecture, philosophy, and backend engine.

## 🧠 What Makes LittleLearn Different?

- It is **not a wrapper** or extension of existing libraries. All modules, layers, and
  training engines are **fully original**, including the custom autodiff engine: `GradientReflector`.
- Built with **raw control in mind**: define layers directly, customize everything, and build
  neural networks without being forced into predefined structures.
- Includes both **high-level APIs** (`Sequential`, `Model`) for convenience and **low-level access**
  for full experimental control.
- Features unique innovations like **node-level gradient clipping**, **inline graph execution tracing**,
  and fully customizable attention mechanisms (e.g., MHA from scratch).

## ⚙️ Core Philosophy:
    Touch Big World By Little Step \n 
    bring whatever your build model ways with ecosystem, are you a type want make model by one instruction ? 
    AutoBuildModel and AutoTransformers(coming soon) can answer you want. are want play with Sequential more give control with layers warapper ? 
    Sequential and ModelbyNode(coming soon) answers your call. and more ekstrem with raw layers witout layers wrapper ? you do it just call 
    layers,loss,optimizers but you have to know how to get gradient and weight of model. or more ekstrem you want interaction with 
    engine grad that is Gradient Reflector ? yes its can use by anything you need, all back end layers its availavle in Gradient Reflector, 


## 📚 Included Ecosystem:

- Deep learning (Dense, attention, LSTM, etc.)
- Classical machine learning
- Auto tools like `AutoBuildModel` and `AutoTransformers` for one-liner construction.
- All powered by `Gradient Reflector` — the gradient engine with unique features like per-node clipping.

## 🔖 Note:

Although inspired by Keras and PyTorch, **LittleLearn is an original framework** built from the ground up.
It aims to provide a new perspective on model building — flexible, introspective, and fully customizable.

Author: 
----------
Candra Alpin Gunawan 
"""
__name__ = "littlelearn"
__author__ = "Candra Alpin Gunawan"
__version__ = "0.1.5"
__license__ = "Apache 2.0"
__realese__ = "28-july-2025"
__email__ = "hinamatsuriairin@gmail.com"
__repo__ = "https://github.com/Airinchan818/LittleLearn"

from . import DeepLearning 
from . import ClassicMachineLearning
from . import preprocessing
from .GradientReflector import GradientReflector

def convert_to_tensor(x) :

    """
    for convert a matriks or array, to tensor that suport Gradient Reflector operation. 

    How to use : 

    import LittleLearn as ll 
    import numpy as np 

    a = np.array([2,4,5,2,1]) \n 
    a= ll.convert_to_tensor(a)
    """

    return GradientReflector(x)

def matmul(matrix_a,matrix_b,transpose_a=False,transpose_b=False):
    """
    high dimention tensor operation . matrix_a and matrix_b parameter for array or tensor,
    transpose_a will transpose last dimention A array like (a,b,c) => (a,c,b) and transpose_b 
    will transpose (a,b,c) => (a,c,b) for matrix_b.\n 
    
    how to use : 

    import LittleLearn as ll 
    import numpy as np 

    a = np.array(([[1,2,3],[2,3,4]]))\n
    b = np.array(([[2,4,2],[5,3,2]]))\n

    c = ll.matmul(a,b,transpose_a=True) 

    """
    try:
        if not isinstance (matrix_a,GradientReflector) :
            matrix_a = GradientReflector(matrix_a)
        return matrix_a.matmul(matrix_b,transpose_a=transpose_a,transpose_b=transpose_b)
    except :
        if matrix_a.shape == matrix_b.shape :
            raise RuntimeError(f"mismatch {matrix_a.shape} vs {matrix_b.shape} you have transpose a one of matriks")

def sqrt(x) :
    """
    for count sqrt tensor. \n
    how to use  : 

    a = 100 

    import LittleLearn as ll 

    ll.sqrt(a)

    """
    if not isinstance(x,GradientReflector) : 
        x = convert_to_tensor(x)
    return x.sqrt()


def dot (matriks_a,matriks_b) :
    """
    function for dot product multiple at tensor \n 
    how to use : \n 
    import LiitleLearn as ll \n
    import numpy as np \n 

    a = np.array([2,3,4])\n
    b = np.array([3,2,4])\n

    ll.dot(q,b)
    """
    if not isinstance (matriks_a,GradientReflector):
        matriks_a = GradientReflector(matriks_a)
    return matriks_a.dot(matriks_b)

def sin(a) :
    if not isinstance(a,GradientReflector) :
        a = GradientReflector(a)
    return a.sin()

def cos (a) :
    if not isinstance(a,GradientReflector):
        a = GradientReflector(a) 
    return a.cos()

def reshape(matriks,shape=()):
    try :
        if not isinstance(matriks,GradientReflector):
            matriks = GradientReflector(matriks)
        return matriks.reshape(shape=shape)
    except :
        if len(shape) == 0:
            raise RuntimeError("you have to gift a new shape for this matriks")
        raise RuntimeError (f"can't reshape at {shape}")

def log(x) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.log()

def exp(x) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.exp()

def sum (x,axis=None,keepdims=False) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.sum(axis=axis,keepdims=keepdims)

def tan(x) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.tan()

def clip(vector_or_matriks,min_vals,max_vals) :
    if not isinstance(vector_or_matriks,GradientReflector) :
        vector_or_matriks = GradientReflector(vector_or_matriks)
    return vector_or_matriks.clip(min_vals=min_vals,max_vals=max_vals)

def pow (tensor_or_scallar,pow_values) :
    if not isinstance(tensor_or_scallar,GradientReflector) :
        tensor_or_scallar = GradientReflector(tensor_or_scallar)
    return tensor_or_scallar.pow(power_values=pow_values)

def ones_tensor (shape=()) :
    from numpy import ones
    if len(shape) == 0 :
        raise RuntimeError(f"shape is {shape} can't make ones data")
    return GradientReflector(ones(shape=shape))

def ones_like_tensor (x) : 
    from numpy import ones_like 
    if len(x.shape) <= 0:
        raise RuntimeError(f"tensor like is {x.shape} can't make ones data ")
    if isinstance(x,GradientReflector) : 
        x = x.tensor 
    return GradientReflector(ones_like(x)) 

def zeros_tensor (shape=()) : 
    from numpy import zeros 
    if len(shape) == 0 :
        raise RuntimeError(f"shape is {shape} can't make zeros data")
    return GradientReflector(zeros(shape=shape))

def zeros_like_tensor(x) :
    from numpy import zeros_like 
    if len(x.shape) <=0 :
        raise RuntimeError(f"tensor like {x.shape} can.t make zeros data")
    return GradientReflector(zeros_like(x))

def random_rand(*args : int) :
    from numpy.random import rand 
    return GradientReflector(rand(*args))

def random_normal(loc=0,std=1,shape=()) : 
    from numpy.random import normal 

    if len(shape) <= 0 :
        raise RuntimeError(f"Error shape is {shape} can't make data")
    return normal(loc=loc,scale=std,size=shape)

def random_uniform (low = -1,hight=1,shape=()) :
    from numpy.random import uniform 

    if len(shape) <= 0 :
        raise RuntimeError(f"error shape is {shape} can't make data")
    return GradientReflector(uniform(low=low,high=hight,size=shape))
