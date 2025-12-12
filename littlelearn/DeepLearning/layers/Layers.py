import numpy as np 
from typing import Literal 
import traceback
from littlelearn.DeepLearning import activations
import littlelearn as ll 

class Dense (ll.DeepLearning.layers.Component):
    """
    Fully Connected (Dense) Layer.

    This layer performs a standard linear transformation of the input:
        output = input · weight + bias

    Parameters
    ----------
    units : int
        Number of output neurons for this layer (i.e., dimensionality of the output space).

    activation : callable or None
        Activation function to apply after the linear transformation. If None, no activation is applied.

    initial_weight : str or callable
        Method for initializing weights. Supported options: "normal", "uniform", or a custom function.

    initial_bias : str or callable
        Method for initializing bias. Options: "zeros", "random", or a custom function.

    use_gradient_reflector : bool
        Whether to enable gradient tracking and update. If False, the layer runs in inference-only mode.

    Attributes
    ----------
    weight : ndarray of shape (input_dim, units)
        The trainable weight matrix of the layer.

    bias : ndarray of shape (1, units) or None
        The bias vector. Only present if bias is enabled.

    Notes
    -----
    - This is a core building block in most neural networks.
    - Often followed by a non-linear activation function.
    - Backpropagation requires `use_gradient_reflector=True` to compute gradients.

    Written by: Candra Alpin Gunawan 
    """

    def __init__(self,units,activation : Literal['relu','softmax','sigmoid','linear'] = None,
                 initial_Weight : Literal ['uniform','normal'] = 'normal' ,
                 inital_bias : Literal ['zeros','random'] = 'zeros',name="Dense",
                 use_Gradient_reflector:Literal[True,False] = True,
                 reduce_grad : Literal['mean','sum'] = 'sum'):
        super().__init__()
        self.units = units 
        self.activation = activation
        self.weight = None 
        self.bias = None 
        self.initial_weight = initial_Weight
        self.initial_bias = inital_bias
        self.__activation_name = None 
        self.parameter = 0
        self.name = name 
        self.output = None 
        self.logits = None 
        self.__Feature_flag = None 
        self.y_label = None
        self.__input = None 
        self.use_grad_ref = use_Gradient_reflector
        self.weights = None 
        self.out_shape = None 
        self.__grad_methode = reduce_grad

    def __softmax (self,x) :
        if not self.use_grad_ref:
            x_max = np.max(x,axis=-1,keepdims=True)
            x_exp = np.exp(x - x_max,dtype=np.float32)
            x_sum = np.sum(x_exp,axis=-1,dtype=np.float32,keepdims=True)
            x_sum[x_sum==0] = 1e-7
            self.output= x_exp / x_sum
            return self.output
        return activations.softmax(x,axis=-1,keepdims=True,use_crossentropy=True)
    
    def get_weight(self) :
        if self.weight is not None :
            return [self.weight,self.bias]
        else :
            return None 

    def __sigmoid (self,x) :
        if not self.use_grad_ref:
            self.output = 1 / (1 + np.exp(-x,dtype=np.float32))
            return self.output
        return activations.sigmoid(x)
        
    def __linear (self,x) :
        if not self.use_grad_ref :
            self.__output = x 
            return self.output
        return activations.linear(x)
        
    def __relu (self,x) :
        if not self.use_grad_ref:
            return   np.maximum(0,x,dtype=np.float32)
        return activations.relu(x)
    
    def __d_relu (self,x) :
        return np.where(x > 0,1,0)
    
    def __d_sigmoid (self,x) :
        return x * (1 - x)
    
    def __d_softmax (self,y) :
        r_d = list()
        for i in range(len(y)) :
            r_d.append(self.output[i] - y[i])
        return np.vstack(r_d)
    
    def __build_weight (self,Features) :
        self.__Feature_flag = Features
        try :
            if self.units <= 0 or self.units is None :
                raise RuntimeError("0 / None is disagreed")
            if self.initial_weight not in ['uniform','normal'] :
                raise RuntimeError("this layers just support for (normal) and uniform")
            if self.initial_bias not in ['zeros','random'] :
                raise RuntimeError("this layers just suport inital bias for zeros and random")
            if self.activation in ['softmax','sigmoid','linear'] or isinstance(self.activation,activations.Sigmoid)\
                or isinstance(self.activation,activations.Softmax) or isinstance(self.activation,activations.Tanh):
                if self.initial_weight == 'normal':
                    normal_xavier = np.sqrt(2 / (Features + self.units))
                    self.weight = np.random.normal(loc=0,scale=normal_xavier,size = (Features,self.units))
                elif self.initial_weight == 'uniform' :
                    uniform_xavier = np.sqrt(6 / (Features + self.units))
                    self.weight = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(Features,self.units))
            else :
                if self.initial_weight == 'normal' : 
                    Normal_He = np.sqrt(2 / Features)
                    self.weight = np.random.normal(loc=0,scale=Normal_He,size=(Features,self.units))
        
                elif self.initial_weight == 'uniform':
                    uniform_He = np.sqrt(6 / Features) 
                    self.weight = np.random.uniform(low=-uniform_He,high=uniform_He,size=(Features,self.units))
            
            if self.activation is None or isinstance(self.activation,activations.Linear):
                if self.initial_weight == 'normal' :
                    Normal_glorot = np.sqrt(2 / (self.units + Features))
                    self.weight = np.random.normal(loc=0,scale=Normal_glorot,size=(Features,self.units))

                elif self.initial_weight == 'uniform' and self.initial_bias == 'random' :
                    uniform_glorot = np.sqrt(6 / (self.units + Features)) 
                    self.weight = np.random.uniform(low=-uniform_glorot,high=uniform_glorot,size=(Features,self.units))
            if self.initial_bias == 'zeros' :
                self.bias = np.zeros((1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight =='normal':
                Normal_He = np.sqrt(2/Features)
                self.bias = np.random.normal(loc=0,scale=Normal_He,size=(1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight=='uniform' :
                uniform_He = np.sqrt(6/Features)
                self.bias = np.random.uniform(low=-uniform_He,high=uniform_He,size=(1,self.units))
            self.parameter = (Features * self.units) + self.unitsS
        except Exception as e :
            if self.units <= 0 or self.units is None :
                e.add_note("recomended for 2 exponential values like = > example = 2,4,8,16,32....N for initial units")
            if self.initial_weight not in ['uniform','normal'] :
                e.add_note("use the available initial methode [normal,uniform]")
            if self.initial_bias not in ['zeros','random'] :
                e.add_note ("use the available initial methode [zeros,random]")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

            
    def __call__ (self,x) :
        if self.weight is None or self.bias is None :
            self.__build_weight(x.shape[-1])
        if x.shape[-1] != self.__Feature_flag :
            self.__build_weight(x.shape[-1])
        self.__input = x 
        if not self.use_grad_ref:
            logits = np.dot(x,self.weight) + self.bias
            self.out_shape = logits.shape
        else : 
            if not isinstance(self.weight,ll.GradientReflector):
                self.weight = ll.GradientReflector(self.weight,_op="WeightDense")
            if not isinstance(self.bias,ll.GradientReflector):
                self.bias = ll.GradientReflector(self.bias,_op="BiasDense")
            if not isinstance(x,ll.GradientReflector) : 
                x = ll.GradientReflector(x,_op='inputDense')
            logits = x.dense_backend(weight=self.weight,bias=self.bias,reduce_grad=self.__grad_methode)
        self.logits = logits
        self.out_shape = logits.shape
        if self.activation is not None :
            try : 
                if self.activation == 'softmax' :
                    return self.__softmax(logits)
                elif self.activation == 'sigmoid' :
                    return self.__sigmoid(logits)
                elif self.activation == 'relu' :
                    return self.__relu(logits)
                elif self.activation == 'linear' :
                    return self.__linear(logits)
            except :
                return self.activation(logits)
        return logits
        
    
    def BackwardPass (self,gradient_values) :
        try :
            if self.use_grad_ref :
                raise RuntimeError("Error : if you use Gradient Reflector all Backpropogation automatic run")
            
            if self.__activation_name is None  :
                if self.activation == 'softmax' :
                    d_logits = self.__d_softmax(self.y_label)
                if self.activation == 'sigmoid' :
                    d_logits = self.__d_sigmoid(self.output)
                if self.activation == 'relu' :
                    d_logits= self.__d_relu(self.logits)
                if self.activation == 'linear' :
                    d_logits = self.logits
                if self.activation is None :
                    d_logits = self.logits
                grad_out = np.zeros_like(gradient_values[0]) 
                if len(gradient_values) > 1:
                    for i in range(gradient_values) :
                        grad_out += gradient_values[i]
                else :
                    grad_out += gradient_values[0]
                grad_next = grad_out * d_logits
                grad_w = np.dot(self.__input.T,grad_next)
                grad_b = np.sum(grad_next,axis=0)
                grad_next = np.dot(grad_next,self.weight.T)
            return {
                'grad_z' : [grad_next],
                'grad_weight' : [grad_w],
                'grad_bias' : [grad_b]
            }
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 



class Attention (ll.DeepLearning.layers.Component):
    """
    Transformer-style Attention Layer (Single Head)

    Implements Scaled Dot-Product Attention as used in Transformer models.

    Formula:
        Attention(Q, K, V) = softmax((Q · Kᵀ) / sqrt(d_k)) · V

    Parameters
    ----------
    units : int
        Output dimensionality for Q, K, and V projections.
    Masking : bool, optional
        Apply causal mask to prevent attending to future tokens. Default is False.
    return_scores_attention : bool, optional
        Return attention weights along with the output. Default is False.
    initial_weight : {'normal', 'uniform'}, optional
        Weight initialization method. Default is 'normal'.
    initial_bias : {'zeros', 'random'}, optional
        Bias initialization method. Default is 'zeros'.
    stop_jacobian : bool, optional
        Disable full Jacobian calculation for softmax. Default is True.
    derivative_func : {'Jacobian', 'Alternative'}, optional
        Choose method for softmax gradient. Default is 'Alternative'.
    use_Gradient_Reflector : bool, optional
        Use external autodiff engine if available. Default is True.

    Attributes
    ----------
    Weight_Q, Weight_K, Weight_V : ndarray or Tensor
        Projection weights.
    bias_q, bias_k, bias_v : ndarray or Tensor
        Projection biases.
    __scores : ndarray
        Attention scores computed during forward pass.

    Methods
    -------
    __call__(Q, K, V)
        Performs forward attention pass.
    BackwardPass(grad_out)
        Computes gradients manually (if use_Gradient_Reflector is False).

    Notes
    -----
    - Supports optional causal masking.
    - Softmax derivative method is switchable.
    - Gradient computation can be handled externally.

    Reference
    ---------
    Vaswani et al., "Attention is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762

    Author: Candra Alpin Gunawan
    """


    def __init__ (self,units,Masking = False,return_scores_attention = False,
                initial_weight : Literal ['normal','uniform'] = 'normal',
                initial_bias : Literal['zeros','random'] = 'zeros',
                stop_jacobian = True ,
                derivative_func : Literal['Jacobian','Alternative'] = 'Alternative',
                use_Gradient_Reflector : Literal[True,False] = True ) :
        super().__init__()
        self.Masking = Masking 
        self.units = units 
        self.Weight_Q = None 
        self.Weight_K = None 
        self.Weight_V = None 
        self.bias_q = None 
        self.bias_k = None 
        self.bias_v = None 
        self.__cache = None 
        self.return_attn_state = return_scores_attention
        self.parameter = 0 
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias
        self.__input = None 
        self.__scores = None  
        self.__mask_vals = None
        self.__key_dims = None  
        self.derivative_mode = derivative_func
        self.use_engine_grad = use_Gradient_Reflector
        self.stop_jacobian = stop_jacobian
        self.name = self.__class__.__name__
        self.out_shape = None 
    
    def __alternative_Derivative_softmax(self,softmax_out,grad_scores) :
        step1 = np.sum(grad_scores * softmax_out,axis=-1,keepdims=True)
        step2 = softmax_out * (grad_scores - step1)
        return step2
        
    
    def __create_masking (self,size) :
        masking = 1 -  np.tril(np.ones(shape=(size,size),dtype=np.float32))
        return masking 
    
    def __softmax (self,x) :
        x_max = np.max(x,axis=-1,keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp,axis=-1,keepdims=True)
        x_sum[x_sum==0] = 1e-9
        return x_exp / x_sum
    
    def __derivative_softmax (self,x,stop_jacobian=True) :
        print("Critical Warning : Jacobian Mode is dangerous , you computer may be crash. ")
        if stop_jacobian is True :
            raise RuntimeError("Jacobian is disabled due to extremely high computation cost. \n Use Alternative_mode instead.")
        batch,seq,dim = x.shape
        grad = np.zeros((batch,seq,dim,dim),dtype=np.float32)
        for b in range(batch) :
            for s in range(seq) :
                y = x[b,s].reshape(-1,1)
                jacobian = np.diagflat(y) - np.dot(y,y.T)
                grad[b,s] = jacobian
        return grad

    def __build_weight (self,features) : 
        try :
            if self.units <= 0 or self.units is None :
                raise RuntimeError("0 / None is disagreed")
            if self.initial_weight not in ['uniform','normal'] :
                raise RuntimeError("Initial methode not available")
            if self.initial_bias not in ['random','zeros'] :
                raise RuntimeError("Initial methode not available")
            if self.initial_weight == 'normal' :
                scales_variance =  np.sqrt(2 / (features + self.units))
                self.Weight_Q = np.random.normal(loc =0 ,scale=scales_variance,size=(features,self.units)) 
                self.Weight_K = np.random.normal(loc=0,scale=scales_variance,size=(features,self.units)) 
                self.Weight_V = np.random.normal(loc=0,scale=scales_variance,size=(features,self.units)) 
            elif self.initial_weight == 'uniform' :
                scales_variance = np.sqrt(6/(features + self.units))
                self.Weight_Q = np.random.uniform(low=-scales_variance,high=scales_variance,size=(features,self.units))
                self.Weight_K = np.random.uniform(low=-scales_variance,high=scales_variance,size=(features,self.units))
                self.Weight_V = np.random.uniform(low=-scales_variance,high=scales_variance,size=(features,self.units))
            if self.initial_bias == 'zeros' :
                self.bias_q = np.zeros((1,self.units),dtype=np.float32)
                self.bias_k = np.zeros((1,self.units),dtype=np.float32)
                self.bias_v = np.zeros((1,self.units),dtype=np.float32)
            if self.initial_bias == 'random' and self.initial_weight == 'normal' :
                scales_variance = np.sqrt(2/(features + self.units))
                self.bias_q = np.random.normal(loc=0,scale=scales_variance,size=(1,self.units))
                self.bias_k = np.random.normal(loc=0,scale=scales_variance,size=(1,self.units))
                self.bias_v = np.random.normal(loc=0,scale=scales_variance,size=(1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight == 'uniform' :
                scales_variance = np.sqrt(2 / (features + self.units))
                self.bias_q = np.random.uniform(low=-scales_variance,high=scales_variance,size=(1,self.units))
                self.bias_k = np.random.uniform(low=-scales_variance,high=scales_variance,size=(1,self.units))
                self.bias_v = np.random.uniform(low=-scales_variance,high=scales_variance,size=(1,self.units))
            weight_param = (features * self.units) * 3 
            bias_param = self.units * 3 
            self.parameter = weight_param + bias_param
        except Exception as e :
            if self.units is None or self.units <= 0 :
                e.add_note("You have initialization units > 0 => (Attention(units= 1 or more))")
                e.add_note("recomended for 2 exponential values like = > example = 2,4,8,16,32....N")
            if self.initial_weight not in ['normal','uniform'] :
                e.add_note("You must choice initial methode normal or uniform")
            if self.initial_bias not in ['random','zeros'] :
                e.add_note("You must choice initial method random or zeros")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def scaled_dot_product (self,Q,K,V,mask = None) :
        key_dim = K.shape[-1]
        self.__key_dims = key_dim
        scores = np.matmul(Q,K.transpose(0,2,1)) / np.sqrt(key_dim)
        if mask is not None :
            scores += (mask * -1e9)
        scores = self.__softmax(scores)
        self.__scores = scores
        result = np.matmul(scores,V)
        return result,scores

    def get_weight(self) :
        return [self.Weight_V,self.Weight_K,self.Weight_V,self.bias_q,self.bias_k,self.bias_v]
    
    def __call__ (self,Query,Keys,Values) :
        
        try :
            if len(Query.shape) != 3 :
                raise RuntimeError("Query shape must be 3 dims (batch,sequence length,dimention layers)")
            elif len(Keys.shape) != 3 :
                raise RuntimeError("Keys shape must be 3 dims (batch,sequence length,dimention layers)")
            elif len(Values.shape) != 3 :
                raise RuntimeError("Values shape must be 3 dims (batch,sequence length,dimention layers)")
            self.__input = [Query,Keys,Values]
            if self.Weight_Q is None or self.bias_q is None \
                or self.Weight_K is None or self.bias_k is None \
                or self.Weight_V is None or self.bias_v is None :
                    self.__build_weight(Query.shape[-1])
            if self.use_engine_grad is False :
                self.__cache = list()
                Q = np.dot(Query,self.Weight_Q) + self.bias_q
                self.__cache.append(Q)
                K = np.dot(Keys,self.Weight_K) + self.bias_k
                self.__cache.append(K)
                V = np.dot(Values,self.Weight_V) + self.bias_v 
                self.__cache.append(V)
                masked = None 
                if self.Masking :
                    masked = self.__create_masking(Q.shape[1])
                    masked = np.expand_dims(masked,axis=0)
                    self.__mask_vals = masked
                outputs,attention_scores = self.scaled_dot_product(Q,K,V,mask=masked)
                if self.return_attn_state :
                    return outputs,attention_scores
                self.out_shape = outputs.shape 
                return outputs
            else :
                if not isinstance(self.Weight_Q,ll.GradientReflector) :
                    self.Weight_Q = ll.convert_to_tensor(self.Weight_Q)
                if not isinstance(self.Weight_K,ll.GradientReflector) :
                    self.Weight_K = ll.convert_to_tensor(self.Weight_K)
                if not isinstance(self.Weight_V,ll.GradientReflector) :
                    self.Weight_V = ll.convert_to_tensor(self.Weight_V)
                if not isinstance(self.bias_q,ll.GradientReflector) :
                    self.bias_q = ll.convert_to_tensor(self.bias_q)
                if not isinstance(self.bias_k,ll.GradientReflector) :
                    self.bias_k = ll.convert_to_tensor(self.bias_k)
                if not isinstance(self.bias_v,ll.GradientReflector) :
                    self.bias_v = ll.convert_to_tensor(self.bias_v)

                if not isinstance(Query,ll.GradientReflector):
                    Query = ll.GradientReflector(Query,_op='Attention')
                out = Query.singlehead_attention_backend(Keys=Keys,Values=Values,
                                                   weight=(self.Weight_Q,self.Weight_K,self.Weight_V),
                                                   bias=(self.bias_q,self.bias_k,self.bias_q),
                                                   use_causal_mask=self.Masking,
                                                   softmax_derivative=self.derivative_mode)
                self.out_shape = out.shape 
                return out 
        except Exception as e :
            if len(Query.shape) != 3 :
                e.add_note(f"look at : Query shape : {Query.shape} is 3 dims ?" )
            if len(Keys.shape) != 3 :
                e.add_note(f"look at : Keys shape : {Keys.shape} is 3 dims ?")
            if len(Values.shape) !=3 :
                e.add_note(f"look at : Values shape : {Values.shape} is 3 dims ?")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def BackwardPass (self,grad_out) :
        try :
            if self.use_engine_grad is True :
                raise RuntimeError("the backward running in Gradient reflector")
            if self.derivative_mode not in ['Jacobian','Alternative'] :
                raise RuntimeError("Derivative Mode not availabel")
            grad = np.zeros_like(grad_out[0])
            if len(grad_out) >1 :
                for i in range(len(grad_out)) :
                    grad+= grad_out[i]
            else :
                grad += grad_out[0]
            d_Values = np.matmul(self.__scores.transpose(0,2,1),grad) 
            d_scores = np.matmul(grad,self.__cache[-1].transpose(0,2,1))
            if self.derivative_mode == 'Jacobian' :
                d_softmax = self.__derivative_softmax(self.__scores,stop_jacobian=self.stop_jacobian)
                d_softmax= np.einsum("bsij,bsj->bsi",d_softmax,d_scores)
            elif self.derivative_mode == 'Alternative' :
                d_softmax = self.__alternative_Derivative_softmax(self.__scores,d_scores)
            if self.Masking is True :
                d_softmax *= (1-self.__mask_vals)
            d_log_q = np.matmul(d_softmax,self.__cache[-2]) / self.__key_dims
            d_log_k = np.matmul(d_softmax,self.__cache[-3]) / self.__key_dims
            grad_wq = np.matmul(self.__input[0].transpose(0,2,1),d_log_q)
            grad_wk = np.matmul(self.__input[1].transpose(0,2,1),d_log_k)
            grad_wv = np.matmul(self.__input[2].transpose(0,2,1),d_Values)
            grad_b_q = np.sum(d_log_q,axis=1)
            grad_b_k = np.sum(d_log_k,axis=1)
            grad_b_v = np.sum(d_Values,axis=1)
            grad_b_q = np.mean(grad_b_q,axis=0,keepdims=True,dtype=np.float32)
            grad_b_k = np.mean(grad_b_k,axis=0,keepdims=True,dtype=np.float32)
            grad_b_v = np.mean(grad_b_v,axis=0,keepdims=True,dtype=np.float32)
            grad_out = [np.dot(d_log_q,self.Weight_Q.T), np.dot(d_log_k,self.Weight_K.T), np.dot(d_Values,self.Weight_V.T)]
            weight_grad = [grad_wq,grad_wk,grad_wv]
            bias_grad = [grad_b_q,grad_b_k,grad_b_v]
            return {
                "grad_z" : grad_out,
                "grad_weight" : weight_grad,
                "grad_bias" : bias_grad
            }
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class MultiHeadAttention (ll.DeepLearning.layers.Component):
    """
    Multi-Head Attention Layer.

    Implements the multi-head attention mechanism from 
    "Attention is All You Need" (Vaswani et al., 2017).

    Projects queries, keys, and values into multiple heads, applies scaled 
    dot-product attention independently, then concatenates the outputs.

    Args:
        units (int): Output dimensions (must be divisible by num_heads).
        num_heads (int): Number of attention heads.
        output_dim (int, optional): Final projection dimension.
        use_casual_mask (bool, optional): If True, applies look-ahead mask.
        return_attention_state (bool, optional): If True, returns attention weights.
        epsilon (float, optional): Small value to stabilize softmax.
        initial_weight (str, optional): Weight init strategy ('normal' or 'uniform').
        initial_bias (str, optional): Bias init strategy ('zeros' or 'random').
        stop_jacobian (bool, optional): If True, disables full Jacobian.
        derivative_func (str, optional): Derivative strategy for softmax.
        use_gradient_reflector (bool, optional): Integrates with autodiff system.

    Attributes:
        parameters (int): Total trainable parameter count.
    
    Written by : Candra Alpin Gunawan 
        
    Reference: Vaswani et al., NeurIPS 2017.
    """

    def __init__ (self,units,num_heads,output_dim = 0,use_causal_mask=False,return_attention_state=False,epsilon=1e-9,
                  initial_weight : Literal['normal','uniform'] = 'normal',
                  initial_bias : Literal ['zeros','random'] = 'zeros',
                  stop_jacobian=True,
                  derivative_func : Literal['Jacobian','Alternative']  = 'Alternative',
                  use_gradient_reflector : Literal[True,False] = True ) :
        super().__init__()
        self.units = units 
        self.parameter = 0 
        self.num_heads = num_heads
        self.key_dims = units // num_heads
        self.epsilon = epsilon 
        self.weight_q = None 
        self.weight_k = None 
        self.weight_v = None 
        self.weight_o = None 
        self.bias_q = None 
        self.bias_k = None 
        self.bias_v = None 
        self.bias_o = None 
        self.__cache = None 
        self.use_casual_mask = use_causal_mask
        self.return_attention = return_attention_state 
        self.outputdim = output_dim
        self.initial_weight = initial_weight 
        self.initial_bias = initial_bias 
        self.derivative_mode = derivative_func
        self.__input = None 
        self.__scores = None 
        self.name = self.__class__.__name__
        self.__Attention_vals = None 
        self.__shape = None 
        self.__Masking_values = None 
        self.use_Engine_grad = use_gradient_reflector
        self.stop_jacobian = stop_jacobian
        self.out_shape = None 

    def __build_weight (self,features) :
        try:
            if self.units <= 0 or self.units is None :
                raise RuntimeError("0 / None  is disagreed ")
            if self.initial_weight not in ['normal','uniform'] :
                raise RuntimeError("initial methode not availabel")
            if self.initial_bias not in ['zeros','random'] :
                raise RuntimeError("initial methode not availabel")
            if self.outputdim < 0  :
                raise RuntimeError("dimention out < 0 is not agreed")
            normal_xavier = np.sqrt(2/(features + self.units))
            uniform_xavier = np.sqrt(6/(features + self.units))
            if self.outputdim is None :
                self.outputdim = 0 
            normal_wo = np.sqrt(2/(self.outputdim + self.units))
            uniform_wo = np.sqrt(6/(self.outputdim + self.units))
            if self.initial_weight == 'normal' :
                self.weight_q = np.random.normal(loc=0,scale=normal_xavier,size=(features,self.units))
                self.weight_k = np.random.normal(loc=0,scale=normal_xavier,size=(features,self.units))
                self.weight_v = np.random.normal(loc=0,scale=normal_xavier,size=(features,self.units))
                if self.outputdim > 0 :
                    self.weight_o = np.random.normal(loc=0,scale=normal_wo,size=(self.units,self.outputdim))
                else : 
                    self.weight_o = np.random.normal(loc=0,scale=normal_wo,size=(self.units,self.units))
            if self.initial_weight == 'uniform' :
                self.weight_q = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(features,self.units))
                self.weight_k = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(features,self.units))
                self.weight_v = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(features,self.units))
                if self.outputdim > 0 :
                    self.weight_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(self.units,self.outputdim))
                else :
                    self.weight_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(self.units,self.units))
            if self.initial_bias == 'zeros' :
                self.bias_q = np.zeros((1,self.units))
                self.bias_k  = np.zeros((1,self.units))
                self.bias_v = np.zeros((1,self.units))
                if self.outputdim == 0 :
                    self.bias_o = np.zeros((1,self.units))
                else :
                    self.bias_o = np.zeros((1,self.outputdim))
            if self.initial_bias == 'random' and self.initial_weight == 'normal' :
                self.bias_q = np.random.normal(loc=0,scale=normal_xavier,size=(1,self.units))
                self.bias_k = np.random.normal(loc=0,scale=normal_xavier,size=(1,self.units))
                self.bias_v = np.random.normal(loc=0,scale=normal_xavier,size=(1,self.units))
                if self.outputdim > 0 :
                    self.bias_o = np.random.normal(loc=0,scale=normal_wo,size=(1,self.outputdim))
                else :
                    self.bias_o = np.random.normal(loc=0,scale=normal_wo,size=(1,self.units))
            if self.initial_bias == 'random' and self.initial_weight == 'uniform' : 
                self.bias_q = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(1,self.units))
                self.bias_k = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(1,self.units))
                self.bias_v = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(1,self.units))
                if self.outputdim > 0 :
                    self.bias_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(1, self.outputdim))
                else :
                    self.bias_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(1,self.units))
            weight_param = (features * self.units) * 3
            bias_param = self.units * 4 
            self.parameter = weight_param + bias_param + (features * self.weight_o.shape[-1])
        except Exception as e :
            if self.units <=0 or self.units is None :
                e.add_note("You have initialization units > 0 => (MultiheadAttention(units= 1 or more))")
                e.add_note("recomended for 2 exponential values like = > example = 2,4,8,16,32....N")
            elif self.outputdim < 0 :
                e.add_note("for output dimention you can give it to 0 or if you want use it give this parameters. example => 16,32,64...etc")
            if self.initial_weight not in ['normal','uniform'] :
                e.add_note("this layers just support normal or uniform initial weight")
            if self.initial_bias not in ['zeros','random'] :
                e.add_note("this layers just support zeros or random initial bias")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def __softmax(self,x) :
        x_max = np.max(x,keepdims=True,axis=-1)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp,keepdims=True,axis=-1)
        x_sum[x_sum==0] = self.epsilon
        return x_exp / x_sum
    
    def __derivative_softmax (self,x,stop_jacobian=True) :
        stop_jacobian = stop_jacobian
        print("Critical Warning : Jacobian Mode is dangerous , you computer may be crash. ")
        if stop_jacobian is True :
            raise RuntimeError("Jacobian is disabled due to extremely high computation cost. \n Use Alternative_mode instead.")
        batch,head,seq,dim = x.shape
        grad = np.zeros((batch,head,seq,dim,dim))
        for b in range(batch) :
            for h in range(head) :
                for s in range(seq) :
                    y = x[b,h,s].reshape(-1,1)
                    jacob = np.diagflat(y) - np.dot(y,y.T)
                    grad[b,h,s] = jacob 
        return grad 
    
    def get_weight(self) : 
        return [self.weight_q,self.weight_k,self.weight_v,self.weight_o,self.bias_q,self.bias_k,self.bias_v,self.bias_o]
    
    def __alternative_derivative_softmax (self,softmax_out,gradient) :
        step1 = np.sum(softmax_out * gradient,axis=-1,keepdims=True)
        step2 = softmax_out * (gradient - step1) 
        return step2 
    
    def __create_look_ahead_masking (self,size) :
        mask = 1 - np.tril(np.ones(shape=(size,size),dtype=np.float32))
        return mask[np.newaxis,np.newaxis,:,:]
    
    def __splitheads (self,x) :
        batch_size = x.shape[0]
        sequence_lenght = x.shape[1]
        x = np.reshape(x,newshape=(batch_size,sequence_lenght,self.num_heads,(self.units // self.num_heads)))
        x = np.transpose(x,axes=[0,2,1,3])
        return x 
    
    def scaled_dot_product (self,Q,K,V,mask=None) :
        scores = np.matmul(Q,K.transpose(0,1,3,2)) / np.sqrt(self.key_dims)
        if mask is not None :
            scores += (mask * -1e9)
        scores = self.__softmax(scores)
        self.__scores = scores
        attention = np.matmul(scores,V)
        return attention,scores
    
    def __call__ (self,Query,Keys,Values) :
        try :
            self.__input = [Query,Keys,Values]
            if len(Query.shape) != 3 :
                raise RuntimeError(f"Error shape Query : {len(Query.shape)}")
            elif len(Keys.shape) != 3 :
                raise RuntimeError(f"Error shape Keys : {len(Keys.shape)}")
            elif len(Values.shape) != 3 :
                raise RuntimeError(f"Error shape Values : {len(Values.shape)}")
            
            if self.weight_q is None or self.bias_q is None\
            or self.weight_k is None or self.bias_k is None\
            or self.weight_v is None or self.bias_q is None\
            or self.weight_o is None or self.bias_o is None :
                self.__build_weight(Query.shape[-1])
            if self.use_Engine_grad is False :
                self.__cache = list()
                Q = np.dot(Query,self.weight_q) + self.bias_q
                K = np.dot(Keys,self.weight_k) + self.bias_k
                V = np.dot(Values,self.weight_v ) + self.bias_v
                self.__shape = Q.shape
                Q = self.__splitheads(Q)
                K = self.__splitheads(K)
                V = self.__splitheads(V)
                self.__cache.append(Q)
                self.__cache.append(K)
                self.__cache.append(V)
                masked = None 
                if self.use_casual_mask :
                    masked = self.__create_look_ahead_masking(Query.shape[-2])
                    masked = np.expand_dims(masked,axis=0)
                    self.__Masking_values = masked
                
                attention,scores = self.scaled_dot_product(Q,K,V,mask=masked)
                attention = np.transpose(attention,axes=[0,2,1,3])
                attention = np.reshape(attention,newshape=(Query.shape[0],Query.shape[1],self.units))
                result = np.dot(attention,self.weight_o) + self.bias_o
                self.__Attention_vals = attention
                self.__cache.append(result)
                self.out_shape = result.shape 
                if self.return_attention :
                    return result,scores
                return result
            else :
                if not isinstance(self.weight_q,ll.GradientReflector) :
                    self.weight_q = ll.convert_to_tensor(self.weight_q)
                if not isinstance(self.weight_k,ll.GradientReflector) :
                    self.weight_k = ll.convert_to_tensor(self.weight_k)
                if not isinstance(self.weight_v,ll.GradientReflector) :
                    self.weight_v = ll.convert_to_tensor(self.weight_v)
                if not isinstance(self.weight_o,ll.GradientReflector) :
                    self.weight_o = ll.convert_to_tensor(self.weight_o)
                if not isinstance(self.bias_q,ll.GradientReflector) :
                    self.bias_q = ll.convert_to_tensor(self.bias_q)
                if not isinstance(self.bias_k,ll.GradientReflector) :
                    self.bias_k = ll.convert_to_tensor(self.bias_k)
                if not isinstance(self.bias_v,ll.GradientReflector) :
                    self.bias_v = ll.convert_to_tensor(self.bias_v)
                if not isinstance(self.bias_o,ll.GradientReflector) :
                    self.bias_o = ll.convert_to_tensor(self.bias_o)
                if not isinstance (Query,ll.GradientReflector):
                    Query = ll.GradientReflector(Query,_op='Query')
                output = Query.multihead_attention_backend(Keys=Keys,Values=Values,
                                                           num_head=self.num_heads,
                                                           weight=(self.weight_q,self.weight_k,self.weight_v,self.weight_o),
                                                           bias=(self.bias_q,self.bias_k,self.bias_v,self.bias_o),
                                                           use_causal_mask=self.use_casual_mask,
                                                           softmax_derivative=self.derivative_mode)
                self.out_shape = output.shape 
                return output
        except Exception as e :
            e.add_note("Input must be 3 dims ")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def BackwardPass (self,gradient) :
        try:
            if self.use_Engine_grad is True :
                raise RuntimeError("All backward proses running in Gradient Reflector")
            if self.derivative_mode not in ['Jacobian','Alternative'] :
                raise RuntimeError("Derivative mode not vailabel")
            batch,seq,dim = self.__shape
            if len(gradient) >1 :
                grad = np.zeros_like(gradient[0])
                for i in range(len(gradient)) :
                    grad += gradient[i]
            else :
                grad = gradient[0]
            d_Wo = np.matmul(self.__Attention_vals.transpose(0,2,1),grad)
            d_attention_f_W_o = np.dot(grad,self.weight_o.T)
            d_attention_f_W_o = self.__splitheads(d_attention_f_W_o)
            d_wv = np.matmul(self.__scores.transpose(0,1,3,2),d_attention_f_W_o)
            d_scores = np.matmul(d_attention_f_W_o,self.__cache[-2].transpose(0,1,3,2))
            if self.derivative_mode == 'Jacobian' :
                grad_softmax = self.__derivative_softmax(self.__scores,stop_jacobian=self.stop_jacobian)
                d_softmax = np.einsum("bhsdj,bhsj->bhsd",grad_softmax,d_scores)
            elif self.derivative_mode == 'Alternative' :
                d_softmax = self.__alternative_derivative_softmax(self.__scores,d_scores)
            if self.use_casual_mask is True :
                d_softmax *= (1 - self.__Masking_values)
            d_q = np.matmul(d_softmax,self.__cache[-3]) / np.sqrt(self.key_dims)
            d_k = np.matmul(d_softmax,self.__cache[-4]) / np.sqrt(self.key_dims)
            d_q = np.transpose(d_q,(0,2,1,3))
            d_k = np.transpose(d_k,(0,2,1,3))
            d_v = np.transpose(d_wv,(0,2,1,3))
            grad_q = np.matmul(self.__input[0].transpose(0,2,1),d_q.reshape(batch,seq,dim))
            grad_k = np.matmul(self.__input[1].transpose(0,2,1),d_k.reshape(batch,seq,dim))
            grad_v = np.matmul(self.__input[2].transpose(0,2,1),d_v.reshape(batch,seq,dim))
            grad_b_q = np.sum(d_q,axis=1)
            grad_b_k = np.sum(d_k,axis=1)
            grad_b_v = np.sum(d_v,axis=1)
            grad_b_o = np.sum(d_Wo,axis=1)
            grad_b_q = np.mean(grad_b_q,axis=0,keepdims=True,dtype=np.float32)
            grad_b_k = np.mean(grad_b_k,axis=0,keepdims=True,dtype=np.float32)
            grad_b_v = np.mean(grad_b_v,axis=0,keepdims=True,dtype=np.float32)
            grad_b_o = np.mean(grad_b_o,axis=0,keepdims=True,dtype=np.float32)
            grad_nq = np.dot(grad_q,self.weight_q.T)
            grad_nk = np.dot(grad_k,self.weight_k.T)
            grad_nv = np.dot(grad_v,self.weight_v.T)
            grad_next = [grad_nq,grad_nk,grad_nv]
            weight_grad = [grad_q.mean(axis=0),grad_k.mean(axis=0),grad_v.mean(axis=0),d_Wo.mean(axis=0)]
            bias_grad = [grad_b_q,grad_b_k,grad_b_v,grad_b_o]
            return {
                "grad_z" :grad_next,
                "grad_weight" : weight_grad,
                "grad_bias" : bias_grad
            }
        except Exception as e :
            e.add_note("choice jacobian or alternative")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 


class Embedding (ll.DeepLearning.layers.Component):
    """
        Embedding Layer

        Description:
        ------------
        This layer maps discrete input tokens (e.g., integer indices) to continuous, trainable vector embeddings.
        It is primarily used in Natural Language Processing (NLP) to transform categorical inputs into dense representations.

        Parameters:
        -----------
        - input_dim : int  
            The size of the vocabulary (i.e., the maximum index + 1).
        - output_dim : int  
            The dimensionality of each embedding vector.
        - initial_weight : {'normal', 'uniform'}, default='normal'  
            Initialization strategy for the embedding matrix.

        Attributes:
        -----------
        - weight : GradientReflector  
            A trainable weight matrix of shape (input_dim, output_dim), wrapped by a custom gradient manager.
        - parameters : int  
            The total number of trainable parameters: input_dim × output_dim.

        Notes:
        ------
        - Input must be an integer tensor representing categorical indices.
        - The embedding matrix is initialized lazily during the first forward call.
        - If `initial_weight` is not one of {'normal', 'uniform'}, a RuntimeError will be raised.
        - Supports integration with external autodiff engines (e.g., Gradient Reflector).
        - Embeddings are fully trainable if gradient tracking is enabled.

        Example:
        --------
        >>> embed = Embedding(input_dim=1000, output_dim=64)
        >>> output = embed(np.array([1, 5, 999]))  # output shape: (3, 64)

        Author:
        -------
        Candra Alpin Gunawan  
        (Custom architecture and implementation)
    """

    def __init__ (self,input_dim,output_dim) :
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = None 
        self.parameter = 0
        self.name = self.__class__.__name__
        self.out_shape = None 
    
    def __build_weight(self) :
        try :
            if self.input_dim == 0 or self.output_dim is None: 
                raise RuntimeError(f"Error : input_dim {self.input_dim} output_dim {self.output_dim}")
            self.weight = np.random.rand(self.input_dim,self.output_dim)
            if not isinstance(self.weight,ll.GradientReflector) :
                self.weight = ll.GradientReflector(self.weight,_op='embeddingweight')
            self.parameter = self.input_dim * self.output_dim
        except Exception as e :
            e.add_note("Units must initilization => Embedding (units)")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    def get_weight(self) :
        return [self.weight]
    
    def __call__ (self,x) :
        if self.weight is None :
            self.__build_weight()
        if not isinstance(x,ll.GradientReflector) :
            x = ll.GradientReflector(x,_dtype=np.int32)
        out = x.embedding_back_end(self.weight)
        self.out_shape = out.shape 
        return out 

    
class SimpleRNN (ll.DeepLearning.layers.Component):
    """
    SimpleRNN Layer (Elman-style recurrent cell)

    Computes: h_t = tanh(Wx_t + Uh_{t-1} + b)

    Parameters
    ----------
    units : int
        Number of output units (hidden state size).

    Attributes
    ----------
    weight_sequence : ndarray
        Input-to-hidden weight matrix.
    weight_hidden : ndarray
        Hidden-to-hidden (recurrent) weight matrix.
    bias : ndarray
        Bias vector.
    h : ndarray
        Hidden state output.

    Notes
    -----
    - Uses `GradientReflector` for gradient tracking.
    - Automatically builds weights on first input call.
    - Input shape: (batch_size, input_dim)
    - Recommended: powers of 2 for `units` (e.g., 8, 16, 32, ...)

    Author: Candra Alpin Gunawan
    """

    def __init__ (self,units) :
        super().__init__()
        self.units = units 
        self.weight_sequence = None 
        self.name = self.__class__.__name__
        self.weight_hidden = None 
        self.bias = None 
        self.h = None 
        self.parameter = 0 
        self.out_shape = None 

    
    def __build_weight (self,features) :
        try : 
            if self.units == 0 or self.units is None :
                raise ValueError("0 is disagreed for layers") 
            scales_Variance = np.sqrt(2 / features)
            self.weight_sequence = np.random.normal(loc=0,scale=scales_Variance,size=(features,self.units))
            self.weight_hidden = np.random.normal(loc=0,scale=scales_Variance,size=(features,self.units))
            self.bias = np.zeros((1,self.units))
            self.weight_sequence = ll.GradientReflector(self.weight_sequence,_op='weightrnn')
            self.weight_hidden = ll.GradientReflector(self.weight_hidden,_op='weighthidden_s')
            self.bias = ll.GradientReflector(self.bias,_op='biasrnn')
            weight_param = (features * self.units) * 2 
            bias_param = self.units 
            self.parameter = weight_param + bias_param 
        except Exception as e :
            e.add_note("You must initialization units for this layers => SimpleRNN(units)")
            e.add_note("Recomended Units Values is 2 exponential => 2,4,6,16,32,...N")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def get_weight(self) :
        return [self.weight_sequence,self.weight_hidden,self.bias]

    def __call__ (self,x) :
        if self.weight_sequence is None or self.weight_hidden is None :
            self.__build_weight(x.shape[-1])
        self.h = np.zeros((x.shape[0],self.units))
        if not isinstance(self.h,ll.GradientReflector) : 
            self.h = ll.convert_to_tensor(self.h)
        if not isinstance(x,ll.GradientReflector) :
            x = ll.convert_to_tensor(x)
        out=x.simple_rnn_backend(self.h,weight=(self.weight_sequence,self.weight_hidden),bias_= self.bias)
        self.out_shape = out.shape 
        return out 
    

class LSTM (ll.DeepLearning.layers.Component):
    """
    Long Short-Term Memory (LSTM) Layer.

    A standard LSTM layer for processing sequential data.
    Internally utilizes the Gradient Reflector backend for all gate computations,
    weight operations, and gradient tracking.

    Parameters
    ----------
    units : int
        Number of hidden units (state size).
    return_state : bool, optional
        Whether to return the final hidden and cell states (default: False).
    return_sequence : bool, optional
        Whether to return the full sequence or only the final output (default: False).
    weight_initial : str, optional
        Weight initialization method: 'normal' or 'uniform' (default: 'normal').
    bias_initial : str, optional
        Bias initialization method: 'zeros' or 'random' (default: 'zeros').

    Notes
    -----
    - Input shape: (batch_size, sequence_length, input_dim).
    - Hidden and cell states are initialized to zero unless otherwise specified.
    - This class interfaces directly with the Gradient Reflector for automatic differentiation.

    Author
    ------
    Candra Alpin Gunawan
    """

    def __init__ (self,units,return_state=False,return_sequence=False,weight_initial : Literal['normal','uniform'] = 'normal',
                  bias_initial : Literal ['zeros','random'] = 'zeros') :
        super().__init__()
        self.units = units 
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.weight_forgot_gate = None 
        self.weight_new_information_gate = None 
        self.weight_output_gate = None 
        self.weight_cell_state = None 
        self.bias_fg = None 
        self.bias_new_in = None 
        self.bias_output = None 
        self.bias_cell = None 
        self.cell_state = None 
        self.hidden_state = None 
        self.name = self.__class__.__name__
        self.parameters = None 
        self.weight_initial = weight_initial
        self.bias_initial = bias_initial
        self.parameter = 0 
        self.out_shape = None 
    
    def get_weight(self) : 
        return [self.weight_forgot_gate,self.bias_fg,self.weight_new_information_gate,self.bias_new_in,
                self.weight_output_gate,self.bias_output,self.weight_cell_state,self.bias_cell]
    
    def __build_weight (self,features) :
        try :
            He_var = features
            features += self.units
            if self.units is None or self.units == 0:
                raise ValueError ("0 / None units is disagreed for layers")
        
            if self.weight_initial not in ['normal','uniform'] :
                raise ValueError("The initial methods not available at layers")
            if self.weight_initial == 'normal' :
                normal_variance = np.sqrt(2/(He_var))
                self.weight_forgot_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_new_information_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_output_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_cell_state = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
            elif self.weight_initial == 'uniform' :
                uniform_variance = np.sqrt(6 / (He_var))
                self.weight_forgot_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                self.weight_new_information_gate = np.random.uniform(low=-uniform_variance,high = uniform_variance,size=(features,self.units))
                self.weight_output_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                self.weight_cell_state = np.random.uniform (low=-uniform_variance,high=uniform_variance,size=(features,self.units))

            if self.bias_initial not in ['zeros','random'] :
                raise ValueError("bias initial is availabel for zeros and random")
            
            if self.bias_initial == 'zeros' :
                self.bias_fg = np.zeros((1,self.units))
                self.bias_new_in = np.zeros((1,self.units))
                self.bias_cell = np.zeros((1,self.units))
                self.bias_output = np.zeros((1,self.units))

            if self.bias_initial == 'random' and self.weight_initial == 'normal' :
                self.bias_fg = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                self.bias_new_in = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                self.bias_output = np.random.normal(loc=0,scale=uniform_variance,size=(1,self.units))
                self.bias_cell = np.random.normal(loc=0,scale=uniform_variance,size=(1,self.units))

            elif self.bias_initial == 'random' and self.weight_initial == 'uniform' :
                self.bias_fg = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_new_in = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_output = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_cell = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                
                weight_param = (features * self.units) *4 
                bias_param = self.units * 4 
                self.parameter = weight_param + bias_param 

        except Exception as e :
            e.add_note("You must initialization units for this layers => LSTM(units)")
            e.add_note("Recomended Units Values is 2 exponential => 2,4,6,16,32,...N")  
            traceback.print_exception(type(e),e,e.__traceback__)
            raise    

    def __call__ (self,x,initial_state=None) : 
        batch_size,sequence,d_model = x.shape
        if self.weight_forgot_gate is None or self.weight_new_information_gate is None\
        or self.weight_output_gate is None or self.weight_cell_state is None :
            self.__build_weight(d_model)

        if self.hidden_state is None or self.cell_state is None :
            self.hidden_state = np.zeros((batch_size,self.units))
            self.cell_state = np.zeros((batch_size,self.units))

        if initial_state is not None  :
            self.hidden_state,self.cell_state = initial_state

        if not isinstance(self.hidden_state,ll.GradientReflector) :
            self.hidden_state = ll.GradientReflector(self.hidden_state,_op='hidden')

        if not isinstance(self.cell_state,ll.GradientReflector) :
            self.cell_state = ll.GradientReflector(self.cell_state,_op='cell')
        
        if not isinstance(x,ll.GradientReflector) :
            x = ll.convert_to_tensor(x)

        if not isinstance(self.weight_forgot_gate,ll.GradientReflector) :
            self.weight_forgot_gate = ll.GradientReflector(self.weight_forgot_gate,_op='w_fg')
        if not isinstance(self.weight_cell_state,ll.GradientReflector) :
            self.weight_cell_state = ll.GradientReflector(self.weight_cell_state,_op='w_cg')
        if not isinstance(self.weight_new_information_gate,ll.GradientReflector) :
            self.weight_new_information_gate = ll.GradientReflector(self.weight_new_information_gate,_op='w_ig')
        if not isinstance(self.weight_output_gate,ll.GradientReflector) :
            self.weight_output_gate = ll.GradientReflector(self.weight_output_gate,_op='w_og')
        
        if not isinstance(self.bias_fg,ll.GradientReflector) :
            self.bias_fg = ll.GradientReflector(self.bias_fg,_op='b_fg')
        if not isinstance(self.bias_new_in,ll.GradientReflector) :
            self.bias_new_in = ll.GradientReflector(self.bias_new_in,_op='b_ig')
        if not isinstance(self.bias_output,ll.GradientReflector) :
            self.bias_output = ll.GradientReflector(self.bias_output,_op='b_og')
        if not isinstance(self.bias_cell,ll.GradientReflector) :
            self.bias_cell = ll.GradientReflector(self.bias_cell,_op='b_cg')

        
        out = x.lstm_backend(hidden_state=self.hidden_state,
                              cell_state=self.cell_state,weight=(
                                  self.weight_forgot_gate,self.weight_new_information_gate,
                                  self.weight_output_gate,self.weight_cell_state
                              ),bias=(
                                  self.bias_fg,self.bias_new_in,
                                  self.bias_output,self.bias_cell
                              ),return_sequence=self.return_sequence,return_state=self.return_state)
        self.out_shape = out.shape 
        return out 
    
        
        
        
class GRU (ll.DeepLearning.layers.Component):
    """
        Gated Recurrent Unit (GRU) Layer.

        This layer implements the GRU mechanism from scratch with custom backend.

        Args:
            units (int): Number of output features (neurons).
            return_sequences (bool): Whether to return the full sequence.
            return_state (bool): Whether to return the final hidden state.
            initial_weight (str): Weight init type, 'normal' or 'uniform'.
            initial_bias (str): Bias init type, 'zeros' or 'random'.

        Notes:
            - Manual implementation, no external library.
            - Backprop enabled via GradientReflector.

        Author:
            Candra Alpin Gunawan
    """


    def __init__(self,units,return_sequence = False,return_hidden_state = False,
                 initial_weight : Literal['normal','uniform'] = 'normal',
                 initial_bias : Literal['zeros','random'] = 'zeros',) :
        super().__init__()
        self.units = units 
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias
        self.return_sequence = return_sequence
        self.return_hidden_state = return_hidden_state
        self.weight_up_gate = None 
        self.weight_re_gate = None 
        self.weight_candidate_gate = None 
        self.bias_up_gate = None 
        self.bias_re_gate = None 
        self.bias_candidate_gate = None  
        self.parameter = 0
        self.hidden_state = None
        self.name = self.__class__.__name__ 
        self.out_shape = None 
    
    def __build_weight (self,features) : 
        try:
            if self.initial_weight not in ['normal','uniform'] :
                raise RuntimeError("initial methode just availabel for normal and uniform")
            if self.initial_bias not in ['random','zeros'] :
                raise RuntimeError("initial bias methode just availabel for zeros and random")
            if self.units <= 0 or self.units is None  :
                raise RuntimeError("The units can't if is 0 or None")
            He_Var = features 
            features += self.units
            normal_variance = np.sqrt(2 / (He_Var))
            uniform_variance = np.sqrt(6 / (He_Var))
            if self.initial_weight == 'normal':
                self.weight_up_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_re_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_candidate_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
            elif self.initial_weight == 'uniform' :
                self.weight_up_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                self.weight_re_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                self.weight_candidate_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
            if self.initial_bias == 'random' and self.initial_weight == 'normal' :
                self.bias_up_gate = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                self.bias_re_gate = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                self.bias_candidate_gate = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight == 'uniform' :
                self.bias_up_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_re_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_candidate_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
            if self.initial_bias == 'zeros' :
                self.bias_re_gate = np.zeros((1,self.units))
                self.bias_up_gate = np.zeros((1,self.units))
                self.bias_candidate_gate = np.zeros((1,self.units))
            self.hidden_state = np.zeros((1,self.units))
            param_weight = (features * self.units) * 3 
            param_bias = self.units * 3 
            self.parameter = param_weight + param_bias
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def get_weight(self) :
        weight = [self.weight_up_gate,self.bias_up_gate,self.weight_re_gate,self.bias_re_gate,
                  self.bias_candidate_gate,self.bias_candidate_gate]
        return weight 

    def __call__(self,x,initial_state = None ) : 
        batch,seq,dim = x.shape
        if self.weight_up_gate is None or self.weight_re_gate is None \
        or self.weight_candidate_gate is None : 
            self.__build_weight(x.shape[-1])
        
        if not isinstance(x,ll.GradientReflector) :
            x = ll.convert_to_tensor(x)
        
        if not isinstance(self.weight_up_gate,ll.GradientReflector) :
            self.weight_up_gate = ll.GradientReflector(self.weight_up_gate,_op='w_update')
        
        if not isinstance(self.weight_re_gate,ll.GradientReflector) :
            self.weight_re_gate = ll.GradientReflector(self.weight_re_gate,_op='w_reset')
        
        if not isinstance(self.weight_candidate_gate,ll.GradientReflector) :
            self.weight_candidate_gate = ll.GradientReflector(self.weight_candidate_gate,_op='w_candidate')
        
        if not isinstance(self.bias_up_gate,ll.GradientReflector) :
            self.bias_up_gate = ll.GradientReflector(self.bias_up_gate,_op='b_update')
        
        if not isinstance(self.bias_re_gate,ll.GradientReflector) :
            self.bias_re_gate = ll.GradientReflector(self.bias_re_gate,_op='b_reset')
        
        if not isinstance(self.bias_candidate_gate,ll.GradientReflector) :
            self.bias_candidate_gate = ll.GradientReflector(self.bias_candidate_gate,_op='b_candidate')
        
        if initial_state is None :
            self.hidden_state = np.zeros((batch,self.units))
        else :
            self.hidden_state = initial_state
        
        if not isinstance(self.hidden_state,ll.GradientReflector) :
            self.hidden_state = ll.GradientReflector(self.hidden_state,_op='hidden_state')
        
        out = x.gru_backend(self.hidden_state,
                             weight=(self.weight_up_gate,self.weight_re_gate,self.weight_candidate_gate),
                             bias=(self.bias_up_gate,self.bias_re_gate,self.bias_candidate_gate),
                             return_sequence=self.return_sequence,
                             return_state=self.return_hidden_state)
        self.out_shape = out.shape 
        return out

class LatenConnectedBlock (ll.DeepLearning.layers.Component):
    """
        submodel block for Laten Connected Model (LCM) Layer
        -----------------------------------------------------
            LatenConnectedModel is experimental model for look laten 
            relatation with relavan values by step1 and step2 in sigmoid activation. 
            this model created to solving loss context, big resource problem from RNN model
            and resource hungry from transformers Model. 
            
            Laten Connected Model look at both step1 and step2 to find relavan values. 
            and look relavan values with Tanh activation as relavan magnitude with do residural 
            connection with input layers.
        

            Parameters:
            
            - units : int  
                Number of neurons in the dense layer.
            - NormMode : (Optional)
                'prenorm' or 'postnorm' (default: 'postnorm')
            - LatenActivation : (Optional)
                'sigmoid','gelu','swish' (default: 'sigmoid'),
            - drop_rate : (float)
                rate of dropout mechanism.
            
            Author:
            
            Candra Alpin Gunawan 

            reference:
            -------------
            Candra Alpin Gunawan "LCM : A Latent-Connected MLP Architecture for Universal Deep Learning with Fast Convergence and low Computational Cost"
            Zenodo 01 November 2025" 
            link:https://zenodo.org/records/17501400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2ZmJmNDg3LWI3MTYtNDVlNy05OWEzLTRiOTZkNGFhOTkzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJiZTNhZjBmMGJmN2NmN2EyNWYyMzRiZWI3MjJkMjcwZCJ9.1Tcsiz_aRDDHbR2MmUdf2MkcUPbyKsI88dRGsv1O3MpA-dxBMk7B4JiSvfwk0RKG9SBzV7WGHY3mnth_iEwhTg 
    """
    def __init__(self,units : int,drop_rate : float = 0.1,
                 NormMode : Literal['prenorm','postnorm'] = 'postnorm',
                 laten_activation : Literal['sigmoid','gelu','swish'] = 'sigmoid') :
        super().__init__()
        self.step1_layers = Dense(units)
        self.step2_layers = Dense(units)
        self.magnitude_layers = Dense(units)
        self.drop_out = ll.DeepLearning.layers.DropOut(rate=drop_rate)
        self.norm = ll.DeepLearning.layers.LayerNormalization()

        self.NormMode = NormMode
        self.latenactivation = laten_activation
        self.name = self.__class__.__name__
        self.node_weight = [
            self.step1_layers,self.step2_layers,
            self.magnitude_layers,self.norm
        ]
    
    def get_weight(self) :
        weight = list()
        for node in self.node_weight :
            wg = node.get_weight()
            if wg is not None :
                for w in wg :
                    weight.append(w)
        return weight

    def __call__ (self,x) :
        if self.NormMode == 'prenorm' :
            values = self.norm(x)
            step1 = self.step1_layers(values)
            step2 = self.step2_layers(values) 
            if self.latenactivation == 'sigmoid' :
                step1 = activations.sigmoid(step1)
                step2 = activations.sigmoid(step2)
            elif self.latenactivation == 'gelu' :
                step1 = activations.gelu(step1)
                step2 = activations.gelu(step2)
            elif self.latenactivation == 'swish' :
                step1 = activations.swish(step1)
                step2 = activations.swish(step2)
            else :
                raise RuntimeError("Activation not available")
            laten = step1 + step2 
            laten = self.drop_out(laten)
            outputs = self.magnitude_layers(laten)
            outputs = activations.tanh(outputs)
            outputs = x + outputs 
            return outputs 
        
        elif self.NormMode == 'postnorm' :
            step1 = self.step1_layers(x)
            step2 = self.step2_layers(x) 
            if self.latenactivation == 'sigmoid' :
                step1 = activations.sigmoid(step1)
                step2 = activations.sigmoid(step2)
            elif self.latenactivation == 'gelu' :
                step1 = activations.gelu(step1)
                step2 = activations.gelu(step2)
            elif self.latenactivation == 'swish' :
                step1 = activations.swish(step1)
                step2 = activations.swish(step2)
            else :
                raise RuntimeError("Activation not available")
            laten = step1 + step2 
            laten = self.drop_out(laten)
            outputs = self.magnitude_layers(laten)
            outputs = activations.tanh(outputs)
            self.outputs = x + outputs
            outputs = self.norm(self.outputs)
            return outputs 
        else :
            raise RuntimeError("NormMode just available for prenorm and postnorm")

class LCTBlock (ll.DeepLearning.layers.Component) :
    def __init__ (self,d_model : int ,num_head : int ,drop_rate : float,causal_mask : bool = False) :
        super().__init__()
        self.attention = MultiHeadAttention(
            units=d_model,num_heads=num_head,use_causal_mask=causal_mask
        ) 
        self.dropout = ll.DeepLearning.layers.DropOut(rate=drop_rate)
        self.LCM = LatenConnectedBlock(
            units=d_model,drop_rate=drop_rate,
            NormMode='prenorm',laten_activation='gelu'
        )
        self.layernorm = ll.DeepLearning.layers.LayerNormalization()
    
    def __call__(self,x):
        norm = self.layernorm(x)
        attn =  self.attention(norm,norm,norm)
        attn = self.dropout(attn)
        x = x + attn 

        x = self.LCM(x)
        return x 