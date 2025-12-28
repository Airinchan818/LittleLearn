import littlelearn as ll
from typing import Literal

Global_engine_grad = ll.GradientReflector()
def relu (x,device : Literal["cpu","gpu"]="cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(data=x,
                      device=device,
                      requires_grad=requires_grad)
    return Global_engine_grad .relu(x)
    

def leaky_relu (x,alpha=1e-6,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .leaky_relu(x,alpha=alpha)

def swish(x,beta=1.0,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .swish(x,beta)

def gelu (x,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    return Global_engine_grad .gelu_tanh(x)

    
def softmax(x,axis=None,keepdims=False,
            use_crossentropy=True,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .softmax(
        x,axis=axis,keepdims=keepdims,
        with_crossentropy=use_crossentropy
    )

def sigmoid (x,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .sigmoid(x)

def tanh (x,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .tanh(x)

def log_sigmoid (x,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad.logsigmoid(x)

def log_softmax(x,axis=None,keepdims=False
                ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .log_softmax(x,axis=axis,keepdims=keepdims)

def hard_max (x,axis=None,keepdims=False
              ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)

    return Global_engine_grad .hardmax(x,axis=axis,keepdims=keepdims)

def hard_sigmoid (x,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .hardsigmoid(x)

def hard_tanh (x,min_value=-1.0,max_value=1.0
               ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad.hardtanh(x,min_val=min_value,max_val=max_value)

def hard_shrink(x,lambda_:float=0.5
                ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad.hardshrink(x,lambda_)
    
def relu6 (x,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad.relu6(x)

def celu (x,alpha : float 
          ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad.celu(x,alpha=alpha)

def selu(x,lambda_=1.0507,alpha=1.673
        ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True ) :

    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad.selu(x,lambda_,alpha)

def softplus (x ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .softplus(x)

def softsign (x ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :

    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
    
    return Global_engine_grad .softsign(x)

def elu (x,alpha = 1.0 ,device : Literal["cpu","gpu"] = "cpu",requires_grad=True) :
    if not isinstance(x,ll.Tensor) :
        x = ll.Tensor(x,device=device,requires_grad=requires_grad)
        
    return Global_engine_grad .elu(x,alpha=alpha)




class Relu :
    """
    Relu (Rectified Linear Unit)
    ----------------------------
    This class implements the ReLU activation function, which is one of the most commonly
    used activation functions in neural networks. It outputs:

        f(x) = max(0, x)

    It introduces non-linearity to the model while preserving positive values and
    zeroing out negative ones.

    Example:
    --------
        relu = Relu()
        output = relu(input_tensor)

    Notes:
    ------
    The input will be converted using `convert_to_tensor` if it is not already an instance
    of `GradientReflector`, enabling support for gradient tracking.

    Author : Candra Alpin Gunawan 
    """

    def __call__ (self,x) :
        """
        Applies the ReLU activation function element-wise.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input tensor or value.

        Returns
        -------
        GradientReflector
            Output after applying ReLU activation.
        """
        return Global_engine_grad .relu(x)
    
class Sigmoid :
    """
    Sigmoid
    -------
    This class implements the Sigmoid activation function, which maps input values to the range (0, 1).
    It is commonly used in binary classification problems and output layers.

    The function is defined as:
        f(x) = 1 / (1 + exp(-x))

    Example:
    --------
        sigmoid = Sigmoid()
        output = sigmoid(input_tensor)

    Notes:
    ------
    The input will be converted using `convert_to_tensor` if it is not already an instance
    of `GradientReflector`, allowing gradient tracking during backpropagation.

    Author : Candra Alpin Gunawan 
    """
    def __call__(self,x) :
        """
        Applies the Sigmoid activation function element-wise.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input tensor or value.

        Returns
        -------
        GradientReflector
            Output after applying Sigmoid activation.
        """
        return Global_engine_grad.sigmoid(x)
    
class Leaky_Relu :
    """
    Leaky_Relu (Leaky Rectified Linear Unit)
    ----------------------------------------
    This class implements the Leaky ReLU activation function, a variant of ReLU that allows
    a small, non-zero gradient when the unit is not active (i.e., for negative inputs).

    The function is defined as:
        f(x) = x           if x > 0
               alpha * x   otherwise

    This helps prevent the "dying ReLU" problem during training.

    Parameters
    ----------
    alpha : float, default=1e-2
        Slope of the function for x < 0. A small positive value (e.g., 0.01).

    Example:
    --------
        lrelu = Leaky_Relu(alpha=0.01)
        output = lrelu(input_tensor)

    Notes:
    ------
    Input will be converted to a `GradientReflector` tensor if it is not already,
    to ensure support for gradient computation.

    Author : Candra Alpin Gunawan 
    """

    def __init__(self,alpha=1e-3):
        self.aplha = alpha 
    
    def __call__ (self,x) :
        """
        Applies the Leaky ReLU activation function element-wise.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input tensor or value.

        Returns
        -------
        GradientReflector
            Output after applying Leaky ReLU activation.
        """
        return Global_engine_grad .leaky_relu(x,self.aplha)

class Swish :
    """
    Swish
    -----
    This class implements the Swish activation function, a smooth, non-monotonic function
    proposed by researchers at Google. It has been shown to outperform ReLU on some tasks.

    The function is defined as:
        f(x) = x * sigmoid(beta * x)

    where `beta` is a trainable or fixed parameter controlling the shape of the curve.

    Parameters
    ----------
    beta : float, default=1.0
        The scaling factor applied inside the sigmoid. A higher value makes Swish
        behave more like ReLU; a lower value makes it smoother.

    Example:
    --------
        swish = Swish(beta=1.0)
        output = swish(input_tensor)

    Notes:
    ------
    If `x` is not an instance of `GradientReflector`, it will be automatically
    converted using `convert_to_tensor` to enable gradient propagation.

    Author : Candra Alpin Gunawan 
    """

    def __init__(self,beta = 1.0):
        self.beta = beta 
    
    def __call__(self,x):
        """
        Applies the Swish activation function element-wise.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input tensor or value.

        Returns
        -------
        GradientReflector
            Output after applying the Swish activation.
        """
        return Global_engine_grad .swish(x,self.beta)

class Gelu :
    """
    Gelu (Gaussian Error Linear Unit - tanh Approximation)
    ----------------------------------------------------------
    This class implements the GELU activation function using the sigmoid-based approximation.
    GELU is a smooth, non-linear activation that has gained popularity in modern architectures
    such as transformers (e.g., BERT, GPT).

    In this implementation, the activation is approximated as:
        GELU(x) ≈ x * 0.5 * tanh(sqrt(2 / 3.14) * (x + 0.044715 * x^3 ))

    This approximation is more efficient than the original definition using the standard
    normal distribution CDF, while still maintaining strong empirical performance.

    Example:
    --------
        gelu = Gelu()
        output = gelu(input_tensor)

    Notes:
    ------
    If the input `x` is not already a `GradientReflector`, it will be automatically converted
    using `convert_to_tensor`, enabling gradient tracking during backpropagation.

    Author : Candra Alpin Gunawan 
    """
    def __call__(self, x) :
        """
        Applies the GELU activation function (tanh approximation) element-wise.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input tensor or value.

        Returns
        -------
        GradientReflector
            Output after applying the GELU activation.
        """
        return Global_engine_grad.gelu_tanh(x)

class Tanh :
    """
    Tanh (Hyperbolic Tangent)
    --------------------------
    This class implements the hyperbolic tangent (tanh) activation function,
    which maps input values to the range (-1, 1). It is a smooth, zero-centered
    non-linear activation function, commonly used in recurrent neural networks (RNNs)
    and older feedforward architectures.

    The function is defined as:
        f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Example:
    --------
        tanh = Tanh()
        output = tanh(input_tensor)

    Notes:
    ------
    If the input is not already an instance of `GradientReflector`, it will be converted
    using `convert_to_tensor`, enabling support for automatic differentiation.

    Author : Candra Alpin Gunawan 
    """
    def __call__(self,x) :
        """
        Applies the tanh activation function element-wise.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input tensor or scalar.

        Returns
        -------
        GradientReflector
            Output after applying tanh activation.
        """
        return Global_engine_grad .tanh(x)

class Softmax :
    """
    Softmax Activation
    -------------------
    This class implements the softmax activation function, commonly used in the
    output layer of classification models to convert raw logits into normalized 
    probability distributions.

    The softmax function is defined as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j))  for all j

    Parameters
    ----------
    use_categorical : bool, default=True
        If True, the output is interpreted as a categorical distribution 
        (e.g., one-hot encoded labels). When set to False, the gradient behavior 
        may change depending on loss usage.

    axis : int, default=-1
        The axis along which softmax is computed. Commonly set to -1 
        (last dimension), e.g., for batch-wise classification.

    keepdims : bool, default=True
        Whether to keep the reduced dimensions after computing softmax. 
        This can be useful for alignment in broadcasting operations.

    epsilon : float, default=1e-6
        A small constant to avoid numerical instability such as division by zero 
        or log(0) during backpropagation.

    Example
    -------
    >>> softmax = Softmax(axis=-1)
    >>> output = softmax(input_tensor)

    Notes
    -----
    If the input is not already wrapped in a `GradientReflector`, it will be
    automatically converted to one to support gradient tracking.

    Author : Candra Alpin Gunawan 
    """

    def __init__ (self,use_categorical=True,axis=-1,keepdims=True,epsilon=1e-6) :
        self.use_categorical = use_categorical
        self.axis = axis 
        self.epsilon = epsilon 
        self.keepdims = keepdims
    
    def __call__ (self,x) :
        """
        Applies the softmax function along the specified axis.

        Parameters
        ----------
        x : array-like or GradientReflector
            Input logits or tensor to be normalized.

        Returns
        -------
        GradientReflector
            Output probabilities (summing to 1 along the specified axis).
        """
        return Global_engine_grad.softmax(x,axis=self.axis,keepdims=self.keepdims,
                                            with_crossentropy=self.use_categorical)
    
class Relu6 :

    def __call__(self,x) :
        return Global_engine_grad .relu6(x)

class Celu :

    def __init__(self,alpha : float = 1.0) :
        self.alpha = alpha 

    def __call__ (self,x) :
        return Global_engine_grad .celu(x,self.alpha)
    

class Selu :
    def __init__ (self,lambda_=1.0507,alpha=1.673) :
        self.lambda_ = lambda_ 
        self.alpha = alpha 

    def __call__ (self,x) :
        return Global_engine_grad .selu(x,lambda_=self.lambda_,alpha=self.alpha)

class Elu :
    def __init__ (self,alpha :float = 1.0) :
        self.alpha = alpha 
    
    def __call__ (self,x) :
        return Global_engine_grad .elu(x,self.alpha)

class LogSoftmax :
    def __init__(self,axis=-1,keepdims=True) :
        self.axis =axis 
        self.keepdims = keepdims
    
    def __call__(self,x) :

        return Global_engine_grad .log_softmax(x,self.axis,self.keepdims)

class LogSigmoid :
    def __call__ (self,x) :
        return Global_engine_grad .logsigmoid(x)

class HardSigmoid :
    def __call__(self,x) :
        Global_engine_grad .hardsigmoid(x)

class HardTanh :
    def __init__ (self,min_val = -1.0,max_val = 1.0) :
        self.min_val = min_val 
        self.max_val = max_val
        
    
    def __call__(self,x) :

        return Global_engine_grad .hardtanh(x,min_val=self.min_val,max_val=self.max_val)

class HardShrink :
    def __init__(self,lambda_=0.5) :
        self.lambda_ = lambda_

    def __call__(self,x) :
        return Global_engine_grad .hardshrink(x,lambda_=self.lambda_)


class Softplus :
    def __call__ (self,x) :
        return Global_engine_grad .softplus(x)
    
class Softsign :

    def __call__ (self,x) :
        return Global_engine_grad .softsign(x)