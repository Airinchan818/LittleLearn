import littlelearn as ll 
from typing import Literal,Optional,Callable
import numpy as np 
import traceback 

class Component :

    """
        Component
        ---------------
        Component is abstract layers template for make custom model or 
        spesial model without need to make some function to updating weight.

        How to use:
        ----------------
            ```
                class MLP(Component):
                    def __init__ (self,num_class):
                        super().__init__()
                        self.layers1 = Dense(32,'relu')
                        self.layers2 = Dense(64,'relu')
                        self.final_out = Dense(num_class)
                    
                    def __call__(self,**kwargs):
    
                        x = self.layers1(x)
                        x = self.layers2(x)
                        x = self.layers3(x)
            ```
        
        Author:
        ---------
        Candra Alpin Gunawan 

        Warning:
        ------------------
        you have call  Component or model before get_weight  
    """

    def __init__(self):
        pass 
        
    
    def parameter(self) :
        class_layers = self.__dict__
        param = list()

        for obj in class_layers.values():
            try :
                if isinstance(obj,ll.GradientReflector):
                    param.append(obj)
                elif isinstance(obj,Component):
                    weight = obj.parameter()
                    for w in weight :
                        param.append(w) 
                else:
                    weight = obj.get_weight()
                    if weight is not None :
                        for w in weight :
                            param.append(w)
            except :
                pass 

        return param 
    
   
    def __call__(self,**kwargs):
        raise NotImplementedError
        

class LayerNormalization (Component):
    """
    Layer Normalization Layer.

    This layer applies layer normalization to the input tensor,
    normalizing across the feature dimension. It scales and shifts the
    normalized output using trainable parameters gamma and beta.

    Internally relies on the GradientReflector backend to manage
    gradient tracking and operation handling.

    Parameters
    ----------
    epsilon : float, optional
        A small constant added to the denominator for numerical stability (default: 1e-6).

    Notes
    -----
    - Input shape: (batch_size, ..., features)
    - The normalization is applied over the last dimension.
    - Gamma and beta parameters are initialized lazily during the first call,
      based on the feature dimension.
    - Supports backpropagation via GradientReflector.

    Author
    ------
    Candra Alpin Gunawan
    """
    def __init__ (self,epsilon=1e-6) :
        super().__init__()
        self.beta = None 
        self.gamma = None 
        self.epsilon = epsilon 
        self.parameter = 0 
        self.name = self.__class__.__name__ 
        self.out_shape = None 
    
    def __build_component (self,features) :
        self.beta = ll.GradientReflector(np.zeros((1,features)),_op='beta')
        self.gamma = ll.GradientReflector(np.ones((1,features)),_op='gamma')
        self.parameter += self.beta.shape[-1] + self.gamma.shape[-1]

    def get_weight(self) :
        return [self.beta,self.gamma]
    
    def __call__(self,x) :
        if self.beta is None or self.gamma is None :
            self.__build_component(x.shape[-1])
        if not isinstance(x,ll.GradientReflector) :
            x = ll.GradientReflector(x)
        out = x.layernormalization_backend(gamma=self.gamma,beta=self.beta,epsilon=self.epsilon)
        self.out_shape = out.shape 
        return out 

class BacthNormalization (Component):
    """

        Batch Normalization Layer.

        Applies batch normalization to stabilize and accelerate training.
        Internally uses the Gradient Reflector backend for all computations.\n
        parameters:
        ----------
        epsilon : float, optional (default=1e-6)
            a small float added to variance to avoid dividing by zero.

        Notes:
        -----
        - Input shape: (batch_size, features)
        - This implementation integrates tightly with the Gradient Reflector system.
        - Weights are initialized during the first forward call based on input shape.

        Author: Candra Alpin Gunawan
    """

    def __init__(self,epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = None 
        self.name = self.__class__.__name__ 
        self.beta = None 
        self.parameter = 0 
        self.out_shape = None 
    
    def __build_component (self,features) :
        self.gamma = ll.GradientReflector(np.ones((1,features)),_op='gamma')
        self.beta = ll.GradientReflector(np.zeros((1,features)),_op='beta')
        self.parameter += self.gamma.shape[-1] + self.beta.shape[-1]
    
    def get_weight(self) :
        return [self.gamma,self.beta]

    def __call__ (self,x) :
        if self.gamma is None or self.beta is None  :
            self.__build_component(x.shape[-1])
        if not isinstance(x,ll.GradientReflector) :
            x = ll.convert_to_tensor(x)
        out = x.bacthnormalization_backend(gamma=self.gamma,beta=self.beta,epsilon=self.epsilon)
        self.out_shape = out.shape 
        return out 
    
class GlobalAveragePooling1D (Component):
    """
    Global Average Pooling 1D Layer.

    Applies average pooling over the temporal (or specified) axis.
    Commonly used to reduce the temporal dimension before Dense layers.

    Parameters
    ----------
    axis : int, optional (default=1)
        Axis over which to compute the mean. Typically the time or sequence axis.

    keepdims : bool, optional (default=False)
        If True, retains reduced dimensions with length 1.

    Behavior
    --------
    - Accepts input of shape (batch_size, timesteps, features) or similar.
    - Reduces the specified axis by averaging all values across it.
    - Can be used after RNNs, CNNs, or other sequential layers to flatten temporal info.

    Returns
    -------
    GradientReflector
        Tensor with the specified axis reduced by mean operation.

    Notes
    -----
    - No learnable parameters.
    - Internally relies on `GradientReflector.global_average_pooling_backend`.

    Example
    -------
    >>> gap = GlobalAveragePooling1D()
    >>> output = gap(x)  # x shape: (32, 10, 64) → output shape: (32, 64) if axis=1

    Author
    ------
    Candra Alpin Gunawan
    """
    def __init__(self,axis=1,Keepdims=False) :
        super().__init__()
        self.axis = axis 
        self.keepdims = Keepdims
        self.parameter = 0 
        self.name = self.__class__.__name__ 
        self.out_shape = None 

    def __call__(self,x) :
        if not isinstance (x,ll.GradientReflector) :
            x = ll.GradientReflector(x)
        out = x.global_average_pooling_backend(axis=self.axis,keepdims=self.keepdims)
        self.out_shape = out.shape 
        return out 
        
    def get_weight(self) :
        return None 

class DropOut (Component):
    """
    DropOut Layer
    --------------------
    Non-trainable regularization layer that randomly disables a fraction of
    input neurons during training to prevent overfitting.

    This layer multiplies the input tensor `x` by a binary mask `m` sampled
    from a Bernoulli distribution with probability `(1 - rate)` of keeping
    each unit active. The output is scaled by `1 / (1 - rate)` to preserve
    the expected activation magnitude.

    Formula (forward):
        out = (m * x) / (1 - rate)

    Formula (backward):
        grad_input = (m / (1 - rate)) * grad_output

    Parameters
    ----------
    rate : float, default=0.1
        Fraction of the input neurons to drop. Must be between 0 and 1.
        Example: rate=0.2 means 20% of the neurons are disabled.

    Attributes
    ----------
    out_shape : tuple or None
        Shape of the output tensor after dropout is applied.

    Methods
    -------
    __call__(x)
        Applies dropout to the input tensor during training.
    get_weight()
        Returns None, since Dropout has no trainable weights.

    Notes
    -----
    - Dropout is active only during training. During inference, all neurons
      are kept active (mask m = 1), so the layer behaves as an identity map.
    - It is a stochastic regularizer; thus repeated calls with the same input
      may yield different outputs during training.

    Example
    -------
    >>> layer = DropOut(rate=0.3)
    >>> y = layer(x)  # Applies dropout mask

    Author:
    ---------
    Candra Alpin Gunawan 
    """
    def __init__ (self,rate : float = 0.1) :
        super().__init__()
        self.rate = rate 
        self.out_shape = None 
        self.name = self.__class__.__name__
        self.parameter = 0
    
    def __call__ (self,x) :
        if not isinstance(x,ll.GradientReflector) :
            x = ll.convert_to_tensor(x)
        out = x.drop_out_backend(rate = self.rate)
        self.out_shape = out.shape
        return out 
    
    def get_weight(self) :
        return None 



def uniform_he (tensor) :
    """
        uniform_he\n
        he initiator with uniform distribution\n 

        parameter:\n 
        tensor:GradientReflector

        output:\n
        tensor with he uniform
    """
    if not isinstance(tensor,ll.GradientReflector):
        tensor = ll.GradientReflector(tensor)
    s = tensor.shape
    if len(s) >=2 :
        shape = s[-2]
        std = np.sqrt(6/shape)
    else :
        std = np.sqrt(6/s[0])
    
    tensor_arr = np.random.uniform(low=-std,high=std,size=tensor.shape)
    tensor.tensor = tensor_arr
    return tensor

    
   

class Component :

    """
        Component
        ---------------
        Component is abstract layers template for make custom model or 
        spesial model without need to make some function to updating weight.

        How to use:
        ----------------
            ```
                class MLP(Component):
                    def __init__ (self,num_class):
                        super().__init__()
                        self.layers1 = Dense(32,'relu')
                        self.layers2 = Dense(64,'relu')
                        self.final_out = Dense(num_class)
                    
                    def __call__(self,**kwargs):
    
                        x = self.layers1(x)
                        x = self.layers2(x)
                        x = self.layers3(x)
            ```
        
        Author:
        ---------
        Candra Alpin Gunawan 

        Warning:
        ------------------
        you have call  Component or model before get_weight  
    """

    def __init__(self):
        pass 
        
    
    def parameter(self) :
        class_layers = self.__dict__
        param = list()

        for obj in class_layers.values():
            try :
                if isinstance(obj,ll.GradientReflector):
                    param.append(obj)
                elif isinstance(obj,Component):
                    weight = obj.parameter()
                    for w in weight :
                        param.append(w) 
                else:
                    weight = obj.get_weight()
                    if weight is not None :
                        for w in weight :
                            param.append(w)
            except :
                pass 

        return param 
    
   
    def __call__(self,**kwargs):
        raise NotImplementedError
            

def normal_glorot (tensor) :
    """
        normal_glorot(tensor)\n
        glorot initiator with normal distribution 

        parameters:\n
        tensor : GradientReflector tensor

        output:\n 
        tensor with normal_glorot initiator

    """
    if not isinstance(tensor,ll.GradientReflector):
        tensor = ll.GradientReflector(tensor)
    s = tensor.shape
    if len(s) >= 2:
        shape = s[-1],s[-2]
        std = np.sqrt(2/(shape[0] + shape[1]))
    else :
        shape = s[0]
        std = np.sqrt(2/shape)
    
    tensor_arr = np.random.normal(loc=0.0,scale=std,size=tensor.shape)
    tensor.tensor = tensor_arr
    return tensor 

def uniform_glorot (tensor) :
    """
        uniform_glorot(tensor)\n
        glorot initiator with uniform distribution 

        parameters:\n
        tensor : GradientReflector tensor

        output:\n 
        tensor with uniform_glorot initiator

    """
    if not isinstance(tensor,ll.GradientReflector):
        tensor = ll.GradientReflector(tensor)
    s = tensor.shape
    if len(s)>=2 :
        shape = s[-1],s[-2]
        std = np.sqrt(6/(shape[0] + shape[1]))
        
    else :
        shape = s[0]
        std = np.sqrt(6/shape)
    
    tensor_arr = np.random.uniform(low=-std,high=std,size=tensor.shape)
    tensor.tensor = tensor_arr
    return tensor 

def normal_he (tensor) :

    """
        normal_he\n
        he initiator with normal distribution\n 

        parameter:\n 
        tensor:GradientReflector

        output:\n
        tensor with he normal
    """
    if not isinstance(tensor,ll.GradientReflector):
        tensor = ll.GradientReflector(tensor)
    s = tensor.shape
    if len(s) >=2 :
        shape = s[-2]
        std = np.sqrt(2/shape)
    else :
        std = np.sqrt(2/s[0])
    
    tensor_arr = np.random.normal(loc=0,scale=std,size=tensor.shape)
    tensor.tensor = tensor_arr
    return tensor 