from littlelearn.GradientReflector.GradientReflector import GradientReflector, Node
import jax
import jax.numpy as jnp
from jax import device_put
from typing import Literal
import random
Global_engine_grad = GradientReflector()
float32 = jnp.float32
float16 = jnp.float16
float64 = jnp.float64
int32   = jnp.int32
int64   = jnp.int64
int16  = jnp.int16
bool    = jnp.bool_



def _get_device(device: Literal["cpu", "gpu"]):
    if device == "cpu":
        return jax.devices("cpu")[0]
    elif device == "gpu":
        gpus = jax.devices("gpu")
        if len(gpus) == 0:
            raise RuntimeError("No GPU available for JAX.")
        return gpus[0]
    else:
        raise ValueError(f"Unknown device: {device}")


class Tensor:
    def __init__(self, data, dtype=float32,
                 device: Literal["cpu", "gpu"] = "cpu",
                 requires_grad=False,_node : Node = None ):
        self.requires_grad = requires_grad
        self.is_param = False 

        self.dtype = dtype
        self.device = device
        self.node = _node

        if not isinstance(data, jnp.ndarray):
            arr = jnp.asarray(data, dtype=self.dtype)
        else:
            arr = data.astype(self.dtype)

        self.tensor = device_put(arr, _get_device(device))

        if self.requires_grad and self.node is None:
            self.node = Node(self.tensor)


            
    def to(self, device: Literal["cpu", "gpu"]):
        self.device = device
        self.tensor = device_put(self.tensor, _get_device(device))
        return self

    def __repr__(self):
         
        return (
            f"Tensor(shape={self.tensor.shape}, dtype={self.dtype}, "
            f"device={self.device}, requires_grad={self.requires_grad})\n"
            f"{self.tensor}\n"
            f"Backwardpass class : {self.node.backward.__class__.__name__ if self.node else None}"
        )

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def grad(self):
        return None if self.node is None else self.node.grad

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def size(self):
        return self.tensor.size
    
    def __add__ (self,other) : 
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.add(self,other)
    
    def __sub__(self,other)  :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.subtract(self,other)
    
    def pow(self,factor) :
        return Global_engine_grad.pow(self,factor=factor)
    
    def __mul__ (self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.multiple(self,other)
    
    def __truediv__ (self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.divide(self,other)
    
    def __pow__(self,factor) :
        return Global_engine_grad.pow(self,factor=factor)
    
    def __neg__(self) :
        return self *-1 
    
    def __radd__ (self,other) : 
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.add(other,self)

    def __rsub__ (self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.subtract(other,self)
    
    def __rtruediv__ (self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.divide(other,self)

    def __rmul__ (self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.multiple(other,self)
    
    def matmul (self,other,tranpose_a=False,transpose_b = False) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.matmul(self,other,transpose_a=tranpose_a,transpose_b=transpose_b)
    
    def sum(self,axis=None,keepdims=False) :
        return Global_engine_grad.sum(self,axis=axis,keepdims=keepdims)
    
    def sumprod (self,other) :
        other = other if isinstance(other,Tensor) else Tensor(other,dtype=self.dtype,
                                                              requires_grad=self.requires_grad,
                                                              device=self.device)
        return Global_engine_grad.sumproduct(self,other)
    
    def mean(self,axis=None,keepdims=False) :
        return Global_engine_grad.mean(self,axis=axis,keepdims=keepdims)
    
    def var (self,axis=None,keepdims=False) :
        return Global_engine_grad.var(self,axis=axis,keepdims=keepdims)

    def std (self,axis=None,keepdims=False,epsilon=1e-6) :
        return Global_engine_grad.std(self,axis=axis,keepdims=keepdims,epsilon=epsilon)

    def exp(self) :
        return Global_engine_grad.exp(self)

    def log(self) :
        return Global_engine_grad.log(self)

    def log2(self) :
        return Global_engine_grad.log2(self)

    def log10(self) :
        return Global_engine_grad.log10(self) 
    
    def log1p(self) :
        return Global_engine_grad.log1p(self)
    
    def reshape(self,newshape : tuple) :
        return Global_engine_grad.reshape(self,newshape=newshape)
    
    def unsquezze (self,axis = None) :
        return Global_engine_grad.unsqueeze(self,axis=axis)
    
    def expand_dims (self,axis=None) :
        return Global_engine_grad.expand_dims(self,axis=axis)
    
    def squezze (self,axis=None) :
        return Global_engine_grad.squeeze(self,axis=axis)
    
    def transpose(self,shape : tuple) :
        return Global_engine_grad.transpose(self,axes=shape)
    
    def sqrt(self) :
        return Global_engine_grad.sqrt(self)
    
    def clip(self,min,max) :
        return Global_engine_grad.clip(self,min_value=min,max_value=max)
    
    def max (self,axis=None,keepdims=False) :
        return Global_engine_grad.max(self,axis=axis,keepdims=keepdims)
    
    def min (self,axis=None,keepdims=False) :
        return Global_engine_grad.min(self,axis=axis,keepdims=keepdims)
    
    
    def flatten (self) :
        return Global_engine_grad.flatten(self)
    
    def where (self,condition,then_x,then_y) :
        return Global_engine_grad.where(condition,then_x,then_y)
    

    def argmax (self,axis = None) :
        return Tensor(jnp.argmax(self.tensor,axis=axis) )
    
    def argmin (self,axis=None) :
        return Tensor(jnp.argmin(self.tensor,axis=axis))
    
    def __getitem__ (self,idx) :
        return Global_engine_grad.getitem(self,idx)
    
    def abs (self) :
        return Global_engine_grad.abs(self)
    
    def pad (self,pad_with,mode='constant',constant_value=0) :
        return Global_engine_grad.pad(self,pad_width=pad_with,mode=mode,
                                      constant_values=constant_value)
    
    def broadcast_to (self,shape: tuple) :
        return Global_engine_grad.broadcast_to(self,shape=shape)
    
    def erf (self) :
        return Global_engine_grad.erf(self)
    
    def sinh (self) :
        return Global_engine_grad.sinh(self)
    
    def cosh (self) :
        return Global_engine_grad.cosh(self)
    
    def tan (self) :
        return Global_engine_grad.tan(self)
    
    def floor (self) :
        return Global_engine_grad.floor(self)
    
    def ceil (self) :
        return Global_engine_grad.ceil(self)
    
    def rint (self) :
        return Global_engine_grad.rint(self)
    
    def reciprocal (self) :
        return Global_engine_grad.reciprocal(self)
    
    def sign (self) :
        return Global_engine_grad.sign(self)
    
    def identity (self) :
        return Global_engine_grad.identity(self)
    
    def concat(self,tensor_b,axis=None) :
        tensor_b = tensor_b if isinstance(tensor_b,Tensor) else Tensor(tensor_b,
                                                                       dtype=self.dtype,
                                                                       device=self.device,
                                                                       requires_grad=self.requires_grad)
        return Global_engine_grad.concat([self,tensor_b],axis=axis)
    
    def logsumexp (self,axis=None,keepdims=False) :
        return Global_engine_grad.logsumexp(self,axis=axis,keepdims=keepdims)
    
    def logaddexp (self,b) :
        b = b if isinstance(b,Tensor) else Tensor(b,dtype=self.dtype,
                                                  device=self.device,
                                                  requires_grad=self.requires_grad) 
        return Global_engine_grad.logaddexp(self,b)
    
    def sin(self) :
        return Global_engine_grad.sin(self)
    
    def cos (self) :
        return Global_engine_grad.cos(self)

    
    def backwardpass (self) :
        if self.node is not None : 
            self.node.backwardpass()
    
    def reset_grad (self) :
        if self.node is not None :
            self.node.reset_grad()
    
    def nonactive_grad (self) :
        self.node = None 
        self.requires_grad = False 
    
    def active_grad (self) :
        if self.node is None :
            self.node = Node(self.tensor,parents=())
        self.requires_grad = True


def randn(shape : tuple,dtype = float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False,
          max_random_seed = 10 ) -> Tensor :
    """
    rand:
    ---------
        Returns a tensor with normally distributed random values.
    parameters:
    -----------
        shape (tuple): Shape of the tensor to be created.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
        max_random_seed (int): Maximum value for random seed generation. Default is 10.
    """
    random_seed = random.randint(0,max_random_seed)
    data = jax.random.normal(jax.random.PRNGKey(random_seed),shape).astype(dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def rand (*args : int) -> Tensor :
    """
    rand:
    ---------
        Returns a tensor with uniformly distributed random values in the range [0, 1).
    parameters:
    -----------
        *args (int): Dimensions of the tensor to be created.
    """

    random_seed = random.randint(0,100)
    data = jax.random.uniform(jax.random.PRNGKey(random_seed),shape=args)
    return Tensor(data)

def uniform (low : float = -1.0,high : float = 1.0, shape : tuple = (), 
             dtype=float32,device : Literal["cpu","gpu"]="cpu", max_random_seed = 10,
             requires_grad = False) :
    """
    uniform:
    ---------
        Returns a tensor with uniformly distributed random values in the specified range [low, high).
    parameters:
    -----------
        low (float): Lower bound of the uniform distribution. Default is -1.0.
        high (float): Upper bound of the uniform distribution. Default is 1.0.
        shape (tuple): Shape of the tensor to be created. Default is ().
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        max_random_seed (int): Maximum value for random seed generation. Default is 10.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    random_keys = random.randint(0,max_random_seed) 
    data = jax.random.uniform(
        key=jax.random.PRNGKey(random_keys),shape=shape,dtype=dtype,
        minval=low,maxval=high)
    
    return Tensor(data,device=device,requires_grad=requires_grad)

def zeros(shape : tuple,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    """
        zeros:
        ---------
            Returns a tensor filled with zeros.
        parameters:
        -----------
            shape (tuple): Shape of the tensor to be created.
            dtype: Data type of the tensor. Default is float32.
            device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
            requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.zeros(shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def ones(shape : tuple,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    """
        ones:
        -------
            Returns a tensor filled with ones.
        parameters:
        -----------
            shape (tuple): Shape of the tensor to be created.
            dtype: Data type of the tensor. Default is float32.
            device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
            requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.ones(shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def arange (start : int, end : int, step : int =1 , dtype = int32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    """
    arange:
    ---------
        Returns a tensor with evenly spaced values within a given interval.
    parameters:
    -----------
        start (int): Start of the interval.
        end (int): End of the interval.
        step (int): Spacing between values. Default is 1.
        dtype: Data type of the tensor. Default is int32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.arange(start,end,step,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def eye (n : int, dtype = float32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    """
    eye:
    ---------
        Returns a 2-D tensor with ones on the diagonal and zeros elsewhere. 
    parameters:
    -----------
        n (int): Number of rows and columns of the square tensor.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.eye(n,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def arange_like (tensor : Tensor, start : int, end : int, step : int =1 , dtype = int32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    """
        arange_like
        ---------
            Returns a tensor with evenly spaced values within a given interval, shaped like the input tensor.
        parameters:
        -----------
            tensor (Tensor): Input tensor to match the shape.
            start (int): Start of the interval.
            end (int): End of the interval.
            step (int): Spacing between values. Default is 1.
            dtype: Data type of the tensor. Default is int32.
            device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
            requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.arange(start,end,step,dtype=dtype)
    data = data.reshape(tensor.shape)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def eye_like (tensor : Tensor, dtype = float32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    """
        eye_like:
        ---------
            Returns a 2-D tensor with ones on the diagonal and zeros elsewhere, shaped like the input tensor.
        parameters:
        -----------
            tensor (Tensor): Input tensor to match the shape.
            dtype: Data type of the tensor. Default is float32.
            device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
            requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.

    """
    n = tensor.shape[0]
    data = jnp.eye(n,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def randn_like (tensor : Tensor,dtype = float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    """
    randn_like:
    ---------
        Returns a tensor with normally distributed random values, shaped like the input tensor.
    parameters:
    -----------
        tensor (Tensor): Input tensor to match the shape.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jax.random.normal(jax.random.PRNGKey(random.randint(0,100)),
                             tensor.shape)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def rand_like (tensor : Tensor) -> Tensor :
    """
    rand_like:
    ---------
        Returns a tensor with uniformly distributed random values in the range [0, 1), shaped like the input tensor.
    parameters:
    -----------
        tensor (Tensor): Input tensor to match the shape.
    """
    data = jax.random.uniform(jax.random.PRNGKey(random.randint(0,100)),shape=tensor.shape)
    return Tensor(data)

def zeros_like (tensor : Tensor,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    """
    zeros_like:
    ---------
        Returns a tensor filled with zeros, shaped like the input tensor.
    parameters:
    -----------
        tensor (Tensor): Input tensor to match the shape.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.zeros(tensor.shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def ones_like (tensor : Tensor,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    """
    ones_like:
    ---------
        Returns a tensor filled with ones, shaped like the input tensor.
    parameters:
    -----------
        tensor (Tensor): Input tensor to match the shape.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.ones(tensor.shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def tril (tensor : Tensor,diagonal=0,dtype=float32,device : Literal["cpu","gpu"]="cpu",
          requires_grad = False) -> Tensor:
    """
    tril:
    ---------

        Returns the lower triangular part of a tensor, zeroing out elements above the specified diagonal.
    parameters:
    -----------
        tensor (Tensor): Input tensor.
        diagonal (int): Diagonal above which to zero elements. Default is 0.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    data = jnp.tril(tensor.tensor,k=diagonal)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)


def binomial (n :float , p : float , shape : tuple ,max_random_keys = 10,
              device = "cpu") :
    """
    binomial:
    ---------
        Returns a tensor with random values drawn from a binomial distribution.
    parameters:
    -----------
        n (float): Number of trials.
        p (float): Probability of success on each trial.
        shape (tuple): Shape of the tensor to be created.
        max_random_keys (int): Maximum value for random seed generation. Default is 10.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
    """
    max_random = random.randint(0,max_random_keys)
    data = jax.random.binomial(
        key=jax.random.PRNGKey(max_random),n=n,p=p,shape=shape
    )
    return Tensor(data=data,device=device)

def top_k (logits : Tensor,top_k = 1) :
    """
    top_k:
    ---------
        Returns the top k highest values and their indices from the input tensor.
    parameters:
    -----------
        logits (Tensor): Input tensor from which to extract top k values.
        top_k (int): Number of top elements to retrieve. Default is 1.

    """
    prob,index = jax.lax.top_k(logits.tensor,top_k)
    return Tensor(prob),Tensor(index,dtype=int32)


def categorical (logits : Tensor,max_random_seed = 10,axis=-1) :
    """
    categorical:
    ---------
        Returns a tensor with random values drawn from a categorical distribution defined by the input logits.
    parameters:
    -----------
        logits (Tensor): Input tensor representing the logits for the categorical distribution.
        max_random_seed (int): Maximum value for random seed generation. Default is 10.
        axis (int): Axis along which to sample. Default is -1.
    """
    keys = jax.random.PRNGKey(random.randint(0,max_random_seed))
    categori = jax.random.categorical(key=keys,logits=logits.tensor,axis=axis)
    return Tensor(categori)

def orthogonal (size : int = 1,dtype=float32,device : Literal["cpu","gpu"]="cpu", max_random_seed = 10,
             requires_grad = False) :
    """
    orthogonal:
    ---------
        Returns a square orthogonal tensor of the specified size.
    Parameters:
    -----------
        size (int): Size of the square tensor to be created. Default is 1.
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        max_random_seed (int): Maximum value for random seed generation. Default is 10.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.    
    """
    keys= jax.random.PRNGKey(random.randint(0,max_random_seed))
    logits = jax.random.normal(keys,shape=(size,size))
    Q,R = jnp.linalg.qr(logits)

    data = jnp.sign(jnp.diag(R))
    data = Q * data
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)


def normal(mean :float = 0.0, std = 1.0,shape : tuple = ()
           ,dtype=float32,device : Literal["cpu","gpu"]="cpu", max_random_seed = 10,
             requires_grad = False) :
    """
    normal:
    ---------
        Returns a tensor with normally distributed random values with specified mean and standard deviation.
    parameters:
    -----------
        mean (float): Mean of the normal distribution. Default is 0.0.
        std (float): Standard deviation of the normal distribution. Default is 1.0.
        shape (tuple): Shape of the tensor to be created. Default is ().
        dtype: Data type of the tensor. Default is float32.
        device (str): Device to store the tensor ('cpu' or 'gpu'). Default is 'cpu'.
        max_random_seed (int): Maximum value for random seed generation. Default is 10.
        requires_grad (bool): If True, gradients will be computed for this tensor. Default is False.
    """
    keys = jax.random.PRNGKey(random.randint(0,max_random_seed))
    data = jax.random.normal(keys,shape=shape)
    data = mean + std * data 
    return Tensor(data=data,dtype=dtype,device=device,
                  requires_grad=requires_grad) 

def entropy (probabilities ,axis = -1,epsilon=1e-6) :
    """
    entropy:
    ---------
        Computes the entropy of a probability distribution.
    parameters:
    -----------
        probabilities (Tensor or array-like): Input tensor representing the probability distribution.
        axis (int): Axis along which to compute the entropy. Default is -1.
        epsilon (float): Small value to avoid log(0). Default is 1e-6.
    """
    if not isinstance (probabilities,Tensor) :
        probabilities = Tensor(probabilities)
    if probabilities.requires_grad :
        probabilities.nonactive_grad()
    return - (probabilities * (probabilities + epsilon).log()).sum(axis=axis)



