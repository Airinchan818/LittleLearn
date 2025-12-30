from littlelearn.DeepLearning.activations import Activations as ac
from littlelearn.tensor import Tensor,float32,rand,uniform,zeros,ones
import littlelearn as ll
from typing import Literal
import jax.numpy as jnp 
import numpy as np 
import math 

class Parameter(Tensor):
    def __init__(self, tensor, dtype=float32):
        
        if isinstance(tensor, Tensor):
            super().__init__(
                data=tensor.tensor,
                dtype=dtype,
                requires_grad=False
              
            )
        else:

            super().__init__(
                data=tensor,
                dtype=dtype,
                requires_grad=False
            )
        
        self.is_param = True
    
    def float_16_param (self) :
        self.tensor = jnp.asarray(self.tensor,dtype=jnp.float16)

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

    def load_state(self,model_path="model.npz"):
        
        data = np.load(model_path)
        for i, param in enumerate(self.parameter()):
            key = f"param{i}"
            if key in data:
                param.tensor = jnp.array(data[key])
        print(f"model load from {model_path}")
    
    def save_state(self, model_path="model.npz"):
        param_dict = {f"param{i}": np.array(p.tensor) for i, p in enumerate(self.parameter())}
        np.savez(model_path, **param_dict)
        print(f"model saved at {model_path}")
    
    def parameter(self) :
        param = list()

        for obj in self.__dict__.values():

            if isinstance(obj,Parameter):
                param.append(obj)
            elif isinstance(obj,Component):
                weight = obj.parameter()
                for w in weight :
                    param.append(w) 
            elif isinstance(obj,list) :
                for ob in obj :
                    if isinstance(ob,Component) :
                        weight = ob.parameter()
                        for w in weight :
                            param.append(w) 


        return param 
    
    def to (self,device : str) :
        print(f"{self.__class__.__name__} to {device}")
        for obj in self.parameter() :

            if isinstance(obj,Parameter) :
                obj.to(device)
            

    
    def train(self,state:bool=True) :
        if state :
            
            for obj in self.parameter() :
                if isinstance(obj,Parameter) :
                    obj.active_grad()
            print(f"name : {self.__class__.__name__ } status : active || requires_grad : {obj.requires_grad}")
        else :
            print("all object still inference Mode")

    def inference (self,state : bool = True) :
        if state :
            
            for obj in self.parameter():
                if isinstance(obj,Parameter) :
                    obj.nonactive_grad()
            print(f"name : {self.__class__.__name__ } status : non active || requires_grad : {obj.requires_grad}")
        else :
            print("all object still train Mode")
    
    def set_param_to_float16 (self) :
        for param in self.parameter():
            if isinstance(param,Parameter) :
                param.float_16_param()

    def forwardpass (self,*args,**kwargs) :
        raise NotImplementedError("Define forwardpass() in this class ") 


    def __call__(self, *args, **kwargs):
        return self.forwardpass(*args,**kwargs) 
    

 
class Linear (Component) :

    def __init__ (self,input_dim,output_dim,add_bias=True,dtype=float32) :
        super().__init__()
        self.dtype = dtype
        glorot = math.sqrt(6/(input_dim + output_dim))
        w = uniform(low=-glorot,high=glorot,shape=(input_dim,output_dim),max_random_seed=1000)
        self.weight = Parameter(w,dtype=self.dtype)
        if add_bias :
            bias = zeros((1,output_dim),dtype=dtype)
            self.bias = Parameter(bias,dtype=dtype)
        else :
            self.bias = None 
    
    def forwardpass(self,x):
        if not isinstance(x,Tensor) :
            raise ValueError("input must be Tensor ")
        if self.bias is not None :
            return ll.matmul(x,self.weight) + self.bias
        
        return ll.matmul(x,self.weight)
    
class Embedding (Component) :
    def __init__(self,vocab_size : int , embedding_dim : int) :
        super().__init__()
        self.weight = Parameter(tensor=rand(vocab_size,embedding_dim))
    
    def forwardpass(self,x):
        if isinstance(x,Tensor) :
            x = x.tensor
      
        return self.weight[x]

class GlobalAveragePooling1D (Component) :
    def __init__(self,axis=1,keepdims=False):
        super().__init__() 
        self.axis = axis 
        self.keepdims = keepdims
    
    def forwardpass(self, x):
        if not isinstance(x,Tensor) :
            raise ValueError("input must be Tensor")
        return ll.mean(x,axis=self.axis,keepdims=self.keepdims)


class Attention (Component) :
    def __init__ (self,embed_dim : int,add_bias : bool = False, return_attention = False ,use_causal_mask = False):
        super().__init__()
        std = math.sqrt(6/(embed_dim * 2 ))
        self.wq = Parameter(tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100))
        self.wk = Parameter(tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100))
        self.wv = Parameter(tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100))
        if add_bias :
            self.bq = Parameter(tensor=zeros(shape=(1,embed_dim)))
            self.bk = Parameter(tensor=zeros(shape=(1,embed_dim)))
            self.bv = Parameter(tensor=zeros(shape=(1,embed_dim)))
        else :
            self.bq = None 
            self.bv = None 
            self.bk = None 
        self.is_causal_mask = use_causal_mask
        self.return_attention = return_attention
        self.__add_bias = add_bias
    
    def __created_causal (self,size,device) :
        return 1 - ll.tril(ones(shape=(size,size)),diagonal=0,device=device)
    
    def __scaled_dot_product(self,Q,K,V,causal_mask = None):
        dim_k = ll.to_tensor(K.shape[-1],device=Q.device)
        logits = ll.matmul(Q,K,transpose_b=True) / ll.sqrt(dim_k)
        if causal_mask is not None :
            logits = logits + causal_mask
        score = ll.DeepLearning.activations.softmax(
            x=logits,axis=-1,keepdims=True,use_crossentropy=False
        )
        out = ll.matmul(score,V)
        if self.return_attention :
            return out,score
        else :
            return out 
    
    def forwardpass(self,query,keys,values) :
        if self.__add_bias  :
            q = ll.matmul(query,self.wq) + self.bq
            k = ll.matmul(keys,self.wk) + self.bk 
            v = ll.matmul(values,self.wv) + self.bv 
        
        else :
            q = ll.matmul(query,self.wq)
            k = ll.matmul(keys,self.wk)
            v = ll.matmul(values,self.wv)
        
        if self.is_causal_mask:
        
            mask = self.__created_causal(q.shape[1],device=q.device) * -1e9

            return self.__scaled_dot_product(q,k,v,mask)
        else :
            return self.__scaled_dot_product(q,k,v,causal_mask=None)



class RMSNorm (Component) :
    def __init__ (self,embed_dim : int,epsilon = 1e-6) :
        super().__init__()
        self.weight = Parameter(tensor=ones(shape=(1,embed_dim)))
        self.eps = epsilon
    
    def forwardpass(self,x) :
        rms = ll.sqrt(ll.mean(a=x**2,axis=-1,keepdims=True) + self.eps)
        rms_x = x / rms 
        return rms_x * self.weight

class Dropout(Component) :
    def __init__(self,rate :float = 0.1) :
        super().__init__()
        self.rate = rate 
    
    def forwardpass(self,x) :
        rate = 1 - self.rate
        m = ll.binomial(1,rate,shape=x.shape,device=x.device)
        x = m * x / rate 
        return x 

class LayerNorm (Component) :
    def __init__ (self,embed_dim : int, epsilon=1e-6) :
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = epsilon
        self.w_gamma = Parameter(tensor=ones(shape=(1,self.embed_dim)))
        self.w_beta = Parameter(tensor=zeros(shape=(1,self.embed_dim)))
    
    def forwardpass(self, x):
        mean = ll.mean(x,axis=-1,keepdims=True)
        var = ll.var(x,axis=-1,keepdims=True)
        shifted1 = x - mean 
        shifted2 = ll.sqrt(var + self.eps)
        return shifted1 / shifted2 * self.w_gamma + self.w_beta

class BatchNorm (Component) :
    def __init__(self,embed_dim : int , epsilon = 1e-6) :
        super().__init__()
        self.embed_dim = embed_dim 
        self.eps = epsilon
        self.w_gamma = Parameter(tensor=ones(shape=(1,self.embed_dim)))
        self.w_beta = Parameter(tensor=zeros(shape=(1,self.embed_dim)))
    
    def forwardpass (self,x) :
        mean = ll.mean(x,axis=0,keepdims=True)
        var = ll.var(x,axis=0,keepdims=True)
        shifted1 = x - mean 
        shifted2 = ll.sqrt(var + self.eps)
        return shifted1 / shifted2 * self.w_gamma + self.w_beta


class SimpleRNN (Component) :
    def __init__ (self,embed_dim : int,return_sequence = False ) :
        super().__init__()
        self.embed_dim = embed_dim
        self.return_sequence = return_sequence 
        std = math.sqrt(6 / (embed_dim * 2))
        self.weight_s = Parameter(tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=0))
        self.weight_h = Parameter(tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=0)) 
        self.bias = Parameter(tensor=zeros(shape=(1,embed_dim))) 
        self.hidden_state = None 
    
    def forwardpass(self,x) :
        B,T,D = x.shape 
        self.hidden_state=zeros(shape=(B,D),requires_grad=False,device=x.device)
        output = list()
        for t in range(T) :
            x_t = x[:,t,:]
            h_new = (ll.matmul(x_t,self.weight_s) + ll.matmul(self.hidden_state,self.weight_h) + self.bias)
            h_new = ll.DeepLearning.activations.tanh(h_new)
            output.append(h_new.reshape((B,1,D)))
            self.hidden_state = h_new
        
        
        output =  ll.concat(output,axis=1)

        if self.return_sequence :
            return output
        else :
            return output[:,-1,:] 
 

class MultiHeadAttention (Component) :
    def __init__ (self,embed_dim : int ,num_head : int =None,add_bias=False,return_attention=False,
                  use_causal_mask = False) :
        super().__init__()
        std = math.sqrt(6/(embed_dim * 2))
        self.return_attention = return_attention
        self.use_causal_mask = use_causal_mask
        self.add_bias = add_bias
        self.wq = Parameter(
            tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100)
        )
        self.wk = Parameter(
             tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100)
        )
        self.wv = Parameter(
             tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100)
        )

        self.wo = Parameter(
            tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=100)
        )

        if add_bias :
            self.bq = Parameter(
                tensor=zeros(shape=(1,embed_dim))
            ) 
            self.bk = Parameter(
                tensor=zeros(shape=(1,embed_dim))
            )
            self.bv = Parameter(
                tensor=zeros(shape=(1,embed_dim))
            )
            self.bo = Parameter(
                tensor=zeros(shape=(1,embed_dim))
            )
        else :
            self.bq = None 
            self.bk = None 
            self.bv = None 
        
        if num_head is None :
            if embed_dim >=64 : 
                num_head = embed_dim // 64
            else :
                num_head = 2 

        self.num_head = num_head
        self.dim_k = embed_dim // num_head

    
    def __create_causal_mask (self,size,device) :
        return 1 - ll.tril(ones(shape=(size,size)),diagonal=0,device=device)
    
        
    
    def __split_head(self,X) :
        B,S,_  = X.shape
        X = ll.reshape(X,(B,S,self.num_head,self.dim_k))
        X = ll.transpose(X,(B,self.num_head,S,self.dim_k))
        return X
    
    def __scaled_dot_product (self,Q,K,V,causal_mask=None):
        dim_k = ll.to_tensor(K.shape[-1],device=Q.device)
        score = ll.matmul(Q,K,transpose_b=True) / ll.sqrt(dim_k)
        if causal_mask is not None :
            score = score + causal_mask
        
        attention = ll.DeepLearning.activations.softmax(
            x=score,axis=-1,keepdims=True,use_crossentropy=False
        )

        output = ll.matmul(attention,V)

        return output,attention

    def forwardpass (self,query,keys,values) :
        B,S,D = query.shape
        if self.add_bias :
            q = self.__split_head(ll.matmul(query,self.wq) + self.bq)
            k = self.__split_head(ll.matmul(keys,self.wk) + self.bk)
            v = self.__split_head(ll.matmul(values,self.wv) + self.bv)
        else :
            q = self.__split_head(ll.matmul(query,self.wq))
            k = self.__split_head(ll.matmul(keys,self.wk))
            v = self.__split_head(ll.matmul(values,self.wv))
        
        if self.use_causal_mask :
            mask = self.__create_causal_mask(size=S,device=q.device) * -1e9
            output,attn = self.__scaled_dot_product(
                Q=q,K=k,V=v,causal_mask=mask
            )
        else :
            output,attn = self.__scaled_dot_product(
                Q=q,K=k,V=v,causal_mask=None
            )

        output = ll.transpose(output,new_shape=(B,S,self.num_head,self.dim_k))
        output = ll.reshape(output,new_shape=(B,S,D))

        if self.add_bias :
            output = ll.matmul(output,self.wo) + self.bo
        
        else :
            output = ll.matmul(output,self.wo)

        if self.return_attention :
            return output,attn
        
        else :
            return output
        
class LSTM (Component) :
    def __init__ (self,embed_dim : int, return_sequence = True,return_state = False) :
        super().__init__()
        std = math.sqrt(float(embed_dim))
        self.w_input = Parameter(
            tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=20)
        )
        self.w_forgot = Parameter(
            tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=20)
        )
        self.w_gate = Parameter(
            tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=20)
        )
        self.w_output = Parameter(
            tensor=uniform(low=-std,high=std,shape=(embed_dim,embed_dim),max_random_seed=20)
        )
        self.wh_in = Parameter(
            tensor=ll.orthogonal(size=embed_dim)
        )
        self.wh_fo = Parameter(
            tensor=ll.orthogonal(size=embed_dim)
        )
        self.wh_ga = Parameter(
            tensor=ll.orthogonal(size=embed_dim)
        )
        self.wh_o = Parameter(
            tensor=ll.orthogonal(size=embed_dim)
        )
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.hidden_state = None 
        self.cell_state = None 
    
    def forwardpass (self,x,state = None) :
        B,T,D = x.shape
        if state is not None :
            self.hidden_state = state[0]
            self.cell_state = state[1]
        else :
            self.hidden_state = zeros(shape=(B,D),requires_grad=x.requires_grad,device=x.device)
            self.cell_state = zeros(shape=(B,D),requires_grad=x.requires_grad,device=x.device)
        output = list()
        for t in range(T) :
            x_t = x[:,t,:]
            self.cell_state = ll.GradientReflector.lstm_cell(
                x_t=x_t,h_prev=self.hidden_state,c_prev=self.cell_state,
                w_gate=(self.w_input,self.w_forgot,self.w_forgot),
                wh_gate=(self.wh_in,self.wh_fo,self.wh_ga)
            ) 
            self.hidden_state = ll.GradientReflector.lstm_hidden(
                x_t=x_t,h_prev=self.hidden_state,cell_state=self.cell_state,
                w_output=self.w_output,wh_output=self.w_output
            )
        
            output.append(self.hidden_state.reshape((B,1,D)))
        output = ll.concat(output,axis=1)
        if self.return_sequence :
            if self.return_state :
                return output,self.hidden_state,self.cell_state
            else :
                return output
        
        else :
            if self.return_state :
                return output[:,-1,:],self.hidden_state,self.cell_state
            else :
                return output[:,-1,:]
            
           

class Sequential (Component) :
    def __init__ (self,component : list) :
        super().__init__()
        self.__component = component
    

    def parameter(self):
        param = []
        for obj in self.__component :
            if isinstance(obj,Component) :
                weight = obj.parameter()
                for w in weight :
                    param.append(w)
        
        return param 

    def forwardpass(self,x) :
        for component in self.__component :
            if isinstance(component,Component) :
                x = component(x)
        
        return x 

class LCMBlock (Component) :
    def __init__(self,embed_dim: int,drop_rate :float = 0.1 ) :
        super().__init__()
        self.step1 = Linear(embed_dim,embed_dim)
        self.step2 = Linear(embed_dim,embed_dim)
        self.norm = RMSNorm(embed_dim)
        self.dropout = Dropout(drop_rate)
        self.magnitude_layers = Linear(embed_dim,embed_dim)

    def forwardpass(self,x) :
        prenorm = self.norm(x)
        step1 = self.step1(prenorm)
        step1 = ll.DeepLearning.activations.gelu(step1)
        step2 = self.step2(prenorm)
        step2 = ll.DeepLearning.activations.gelu(step2)
        laten = step1 + step2 
        laten = self.dropout(laten)
        laten = self.magnitude_layers(laten)
        laten = ll.DeepLearning.activations.tanh(laten)
        return x + laten 

class LCTBlock (Component) :
    def __init__ (self,embed_dim : int ,num_head : int,drop_rate : float = 0.1,
                  use_causal_mask = False) :
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim,num_head=num_head,use_causal_mask=use_causal_mask)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.dropout = Dropout(drop_rate)
        self.lcm = LCMBlock(embed_dim,drop_rate)
    
    def forwardpass (self,x) :
        norm1 = self.norm1(x)
        attn = self.attention(norm1,norm1,norm1)
        attn = self.dropout(attn)
        x = x + attn 

        norm2 = self.norm2(x)
        lcm = self.lcm(norm2)
        
        return lcm 


class FeedForwardNetwork (Component) :
    def __init__(self,embed_dim : int ,ffn_dim : int):
        super().__init__()
        self.linear1 = Linear(embed_dim,ffn_dim)
        self.gelu = ll.DeepLearning.activations.Gelu()
        self.linear2 = Linear(ffn_dim,embed_dim) 
    
    def forwardpass(self,x) :
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x 


class TransformersBlock (Component) :
    def __init__(self,embed_dim : int,ffn_dim : int,drop_rate : float = 0.1,type : Literal ['single','multi']= 'multi',
                 mode : Literal['Encoder','Decoder'] = 'Encoder') :
        super().__init__()
        if type == 'multi' :
            if mode == 'Encoder':
                self.attention = MultiHeadAttention(embed_dim=embed_dim)
            elif mode == 'Decoder':
                self.attention = MultiHeadAttention(embed_dim=embed_dim,use_causal_mask=True)
            else :
                raise ValueError(f"{type} not  supported")
        elif type == 'single':
            if mode == 'Encoder' :
                self.attention = Attention(embed_dim=embed_dim)
            elif mode == 'Decoder':
                self.attention = Attention(embed_dim=embed_dim,use_causal_mask=True)
            else :
                raise ValueError(f"{type} not  supported")
        else :
            raise ValueError(f"{type} not supported") 
        
        self.ffn = FeedForwardNetwork(embed_dim,ffn_dim)

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.drop_out1 = Dropout(drop_rate)
        self.drop_out2 = Dropout(drop_rate)
    
    
    def forwardpass(self,x) :
        norm1 = self.norm1(x)
        attn = self.attention(norm1,norm1,norm1)
        attn = self.drop_out1(attn)
        x = x + attn 

        norm2 = self.norm2(x) 
        ffn = self.ffn(norm2)
        x = self.drop_out2(x)
        x = x + ffn 

        return x

class DiagonalSSM(Component):
    def __init__(self, embed_dim: int):
        super().__init__()

 
        d_state = embed_dim

        self.state_w = Parameter(
            tensor=ll.DeepLearning.activations.tanh(
                ll.randn(shape=(d_state,), max_random_seed=30)
            )
        )

        self.wb = Parameter(
            tensor=ll.randn(shape=(d_state, embed_dim), max_random_seed=30)
        )

        self.wc = Parameter(
            tensor=ll.randn(shape=(embed_dim, d_state), max_random_seed=30)
        )

        self.Bd = Parameter(
            tensor=ll.randn(shape=(embed_dim,))
        )

        self.wg = Parameter(
            tensor=ll.randn(shape=(embed_dim, d_state), max_random_seed=30)
        )

    def forwardpass(self, u):
        B, T, D = u.shape
        d_state = self.state_w.shape[0]

        x = zeros((B, d_state), device=u.device)
        output = []

        for t in range(T):
            u_t = u[:, t]

            
            gate = ll.DeepLearning.activations.sigmoid(
                ll.matmul(u_t, self.wg, transpose_b=True)
            ) 


            x_hat = self.state_w * x + ll.matmul(
                u_t, self.wb, transpose_b=True
            )


            x = gate * x_hat + (1.0 - gate) * x


            y = (
                ll.matmul(x, self.wc, transpose_b=True)
                + u_t * self.Bd
                + u_t
            )

            output.append(y.reshape((B, 1, D)))

        return ll.concat(output, axis=1)

class Conv2d (Component) :
    def __init__(self,input_chanel : int , output_chanel : int,kernel_2n : int =2 ,
                 stride : int = 1 ,padding=0,add_bias = False):
        super().__init__()
        std = math.sqrt(6/(input_chanel + output_chanel))
        self.weight = Parameter(
            tensor= uniform(
            low = -std,high=std,shape=(output_chanel,input_chanel,kernel_2n,kernel_2n),
            max_random_seed=100
            ) 
        ) 
        
        if add_bias :
            self.bias = Parameter(
                tensor=zeros(
                    shape=(1,output_chanel)
                )
            )
        else :
            self.bias = None 
        
        self.stride = stride
        self.padding = padding
    
    def forwardpass(self,x) :
        return ll.GradientReflector.conv2d(
            x = x,w=self.weight,b=self.bias,stride=self.stride,pad=self.padding
        )

class GlobalAveragePooling2d (Component) :
    def __init__ (self,axis=(-1,-2),keepdims=False) :
        super().__init__()
        if len(axis) != 2 :
            raise ValueError(f"{axis} error this layers just support 2d axis = (axis1,axis2)")
        self.axis = axis 
        self.keepdims = keepdims
    
    def forwardpass(self,x) :
        return ll.mean(x,axis=self.axis,keepdims=self.keepdims) 

class Maxpool2d (Component) :
    def __init__(self,kernel_2t=2,stride_t=2,padding=0) :
        super().__init__()
        self.kernel = (kernel_2t,kernel_2t)
        self.stride = (stride_t,stride_t)
        self.padding = padding
    
    def forwardpass(self,x) :
        return ll.GradientReflector.maxpool2d(
            x=x,kernel=self.kernel,stride=self.stride,
            padding=self.padding
        )
