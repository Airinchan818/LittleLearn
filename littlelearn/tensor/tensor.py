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
            f"{self.tensor}"
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
    random_seed = random.randint(0,max_random_seed)
    data = jax.random.normal(jax.random.PRNGKey(random_seed),shape).astype(dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def rand (*args : int) -> Tensor :
    random_seed = random.randint(0,100)
    data = jax.random.uniform(jax.random.PRNGKey(random_seed),shape=args)
    return Tensor(data)

def uniform (low : float = -1.0,high : float = 1.0, shape : tuple = (), 
             dtype=float32,device : Literal["cpu","gpu"]="cpu", max_random_seed = 10,
             requires_grad = False) :
    random_keys = random.randint(0,max_random_seed) 
    data = jax.random.uniform(
        key=jax.random.PRNGKey(random_keys),shape=shape,dtype=dtype,
        minval=low,maxval=high)
    
    return Tensor(data,device=device,requires_grad=requires_grad)

def zeros(shape : tuple,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    data = jnp.zeros(shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def ones(shape : tuple,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    data = jnp.ones(shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def arange (start : int, end : int, step : int =1 , dtype = int32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    data = jnp.arange(start,end,step,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def eye (n : int, dtype = float32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    data = jnp.eye(n,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def arange_like (tensor : Tensor, start : int, end : int, step : int =1 , dtype = int32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    data = jnp.arange(start,end,step,dtype=dtype)
    data = data.reshape(tensor.shape)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def eye_like (tensor : Tensor, dtype = float32, device : Literal["cpu","gpu"] ="cpu", requires_grad = False) -> Tensor :
    n = tensor.shape[0]
    data = jnp.eye(n,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def randn_like (tensor : Tensor,dtype = float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    data = jax.random.normal(jax.random.PRNGKey(random.randint(0,100)),
                             tensor.shape)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def rand_like (tensor : Tensor) -> Tensor :
    data = jax.random.uniform(jax.random.PRNGKey(random.randint(0,100)),shape=tensor.shape)
    return Tensor(data)

def zeros_like (tensor : Tensor,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    data = jnp.zeros(tensor.shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def ones_like (tensor : Tensor,dtype=float32,device : Literal["cpu","gpu"] = "cpu",requires_grad = False) -> Tensor :
    data = jnp.ones(tensor.shape,dtype=dtype)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)

def tril (tensor : Tensor,diagonal=0,dtype=float32,device : Literal["cpu","gpu"]="cpu",
          requires_grad = False) -> Tensor:
    data = jnp.tril(tensor.tensor,k=diagonal)
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)


def binomial (n :float , p : float , shape : tuple ,max_random_keys = 10,
              device = "cpu") :
    max_random = random.randint(0,max_random_keys)
    data = jax.random.binomial(
        key=jax.random.PRNGKey(max_random),n=n,p=p,shape=shape
    )
    return Tensor(data=data,device=device)

def top_k (logits : Tensor,top_k = 1) :
    logits = jax.lax.top_k(logits.tensor,top_k)
    return Tensor(logits)


def categorical (logits : Tensor,max_random_seed = 10,axis=-1) :
    keys = jax.random.PRNGKey(random.randint(0,max_random_seed))
    categori = jax.random.categorical(key=keys,logits=logits.tensor,axis=axis)
    return Tensor(categori)

def orthogonal (size : int = 1,dtype=float32,device : Literal["cpu","gpu"]="cpu", max_random_seed = 10,
             requires_grad = False) :
    keys= jax.random.PRNGKey(random.randint(0,max_random_seed))
    logits = jax.random.normal(keys,shape=(size,size))
    Q,R = jnp.linalg.qr(logits)

    data = jnp.sign(jnp.diag(R))
    data = Q * data
    return Tensor(data,dtype=dtype,device=device,requires_grad=requires_grad)


def normal(mean :float = 0.0, std = 1.0,shape : tuple = ()
           ,dtype=float32,device : Literal["cpu","gpu"]="cpu", max_random_seed = 10,
             requires_grad = False) :
    keys = jax.random.PRNGKey(random.randint(0,max_random_seed))
    data = jax.random.normal(keys,shape=shape)
    data = mean + std * data 
    return Tensor(data=data,dtype=dtype,device=device,
                  requires_grad=requires_grad) 