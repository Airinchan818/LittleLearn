import numpy as np 
import traceback          
import matplotlib.pyplot as plt 
import networkx as nx  
from typing import Literal
from numba import jit
import gc
from contextlib import contextmanager
import weakref

active_grad_status = True

@contextmanager 
def non_active_grad () :
    """
        non_active_grad
        ---------------
        call it when a model or another operation not need grad. 

        how to use
        ----------

        from littlelearn import non_active_grad

        with non_active_grad():
            outputs = model(x)
            print(outputs)
        
        Warning!!:
        ---------
        when you want use plot_trace_operation from GradientReflector object or tensor object
        don't use it.
            
        author: Candra Alpin Gunawan 
    """

    global active_grad_status
    active_grad_status = False 
    yield 
    active_grad_status = True 
 
class GradientReflector :
    """
       
        # 📘 GradientReflector - "Reflecting the Flow of Learning"

        Inspired by Andrej Karpathy’s *Micrograd*, `GradientReflector` is a core component of the *LittleLearn* ecosystem —
        a dynamic automatic differentiation engine tailored for modern, vectorized deep learning workflows.

        It acts as both a computation unit and a gradient container, tracking operations in real-time and reflecting their influence during backpropagation.

        ## 🔑 Key Features:

        - 🔁 **Dynamic Computation Graph**:  
        Gradients are computed via reverse-mode autodiff by traversing the computation graph in reverse order.  
        Every operation (`+`, `*`, `**`, etc.) is tracked and contributes to the final gradient.

        - 📐 **Vectorized NumPy Support**:  
        Handles operations on tensors with native NumPy acceleration:  
        `matmul`, `softmax`, `relu`, `reshape`, broadcasting, and more.

        - 📊 **Scalar & Tensor Compatible**:  
        Seamlessly switches between scalar operations (as in Micrograd) and high-dimensional tensor ops (as in deep networks).

        - 🧯 **Auto Gradient Clipping (Auto Node-wise Clipping)**:  
        Built-in gradient explosion protection:
        AutoClipGradient Work to diagnostig any gradient when the all operation 
        do Backpropogation task, its just work when the gradient values its big more than 
        threshold values, and choice clip level protocoll for clipping gradient.

        - 🧼 **Reset Gradients with `kill_grad()`**:  
        Reset gradients across all nodes before re-backwarding.
        Prevents gradient accumulation between training steps.

        - 🔌 **Full Integration with EngineGrad**:  
        When used inside the training engine (`EngineGrad`), `GradientReflector` nodes automatically:
        - Receive clipped gradients
        - Reflect forward & backward computations per layer
        - Participate in auto clipping, loss tracking, and parameter update scheduling

        ## 🔧 Example Usage:

            ```python
            from LittleLearn import GradientReflector

            a = ll.GradientReflector(np.random.rand(10,32))
            b = ll.GradientReflector(np.random.rand(32,64))
            y = a.dot(b)
            y.relu()

            loss = y.sum()
            loss.backwardpass()
            Philosophy Behind the Name:

        GradientReflector is named to emphasize two principles:

        It reflects the computation — all operations are stored and traceable.

        It reflects gradients back — exactly in reverse, for learning to happen.

        Author: Candra Alpin Gunawan 
        """

    def __init__ (self,tensor,_children=(),_op="",_dtype=np.float32) :
        self.tensor = np.array(tensor,dtype=_dtype)
        self.gradient =  np.zeros_like(self.tensor,dtype=_dtype)
        self.active_grad = active_grad_status
        self._backwardpass = lambda : None 
        self._Node = [weakref.ref(child) for child in _children]
        self._op = _op 
        self.__grad_expload_signal = None
        self.__norm_signal = None 
        self.__auto_clip = False 
        self.dtype = _dtype
        self.__autoclip_log = False 
    
    def __repr__(self):
        return (f"(Tensor with shape : ({self.tensor.shape}) : \n  {self.tensor})")
    
    def get_gradient (self) :
        return self.gradient
    
    def __adjust_gradient(self, out_grad, input_grad_shape):
        
        while out_grad.ndim > len(input_grad_shape):
            out_grad = np.sum(out_grad, axis=0)
        while out_grad.ndim < len(input_grad_shape):
            out_grad = np.expand_dims(out_grad, axis=0)
        for i in range(len(input_grad_shape)):
            if input_grad_shape[i] == 1 and out_grad.shape[i] != 1:
                out_grad = np.sum(out_grad, axis=i, keepdims=True)

        return out_grad


    def __add__(self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        if self.active_grad is True :
            out = GradientReflector((self.tensor + other.tensor),(self,other),'+')
        else :
            out = GradientReflector((self.tensor + other.tensor),_op='=')
        def _backward () :
            grad = self.__adjust_gradient(out.gradient,self.tensor.shape)
            grad_other = self.__adjust_gradient(out.gradient,other.tensor.shape)
            self.gradient += grad 
            other.gradient += grad_other
        out._backwardpass = _backward
        return out 
    
    def __sub__ (self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        
        if self.active_grad is True :
            out = GradientReflector((self.tensor - other.tensor),(self,other),'-')
        else :
            out = GradientReflector((self.tensor - other.tensor),_op='-')
        def _backward () :
            grad = self.__adjust_gradient(out.gradient, self.tensor.shape)
            grad_other = self.__adjust_gradient(out.gradient, other.tensor.shape)
            self.gradient += grad
            other.gradient += -grad_other

        out._backwardpass = _backward
        return out 
    
    def pow (self,power_values) :

        if self.active_grad is True :
            out = GradientReflector(np.power(self.tensor,power_values),(self,),'pow')
        else :
            out = GradientReflector(np.power(self.tensor,power_values),_op='pow')
        def _backward() :
            grad = power_values * (np.power(self.tensor,power_values-1))
            self.gradient += grad * out.gradient
        out._backwardpass = _backward
        return out  
    
    def __mul__ (self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        if self.active_grad is True :
            out = GradientReflector((self.tensor * other.tensor),(self,other),_op='*')
        else :
            out = GradientReflector((self.tensor * other.tensor),'*')
        def _backward () :
            grad_= out.gradient
            grad_out = other.tensor * grad_
            grad = self.__adjust_gradient(grad_out,self.gradient.shape)
            self.gradient += grad 
            
            other_grad = self.tensor * grad_
            other_grad = self.__adjust_gradient(other_grad,other.tensor.shape)
            other.gradient += other_grad
        out._backwardpass = _backward
        return out 
    
    def __truediv__(self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other) 
        if self.active_grad is True :
            out = GradientReflector((self.tensor / other.tensor),(self,other),"/")
        else : 
            GradientReflector((self.tensor / other.tensor),_op="/")
        def _backward () :
            grad = self.__adjust_gradient(out.gradient,self.gradient.shape)
            self.gradient += (np.ones_like(self.tensor) / other.tensor) * grad
            other.gradient += (-self.tensor / np.power(other.tensor,2)) * grad 
        out._backwardpass = _backward
        return out 
    
    def __neg__ (self) :
        return self *-1
    
    def __pow__ (self,other) :
        assert isinstance(other,(int,float))
        if self.active_grad is True :
            out = GradientReflector(np.power(self.tensor,other),(self,),f'**{other}') 
        else :
            out = GradientReflector(np.power(self.tensor,other),_op=f'**{other}') 
        def _backward() :
            self.gradient += (other * np.power(self.tensor,other -1)) * out.gradient
        out._backwardpass = _backward
        return out 
    
    def __radd__ (self,other) :
        return self + other 
    
    def __rsub__ (self,other) :
        return self - other 
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, GradientReflector) else GradientReflector(other)
        return other / self


    def __rmul__ (self,other) :
        return self * other 
    
    def relu (self) :
        if self.active_grad is True :
            out = GradientReflector(np.maximum(0,self.tensor),(self,),'relu')
        else : 
            out = GradientReflector(np.maximum(0,self.tensor),_op='relu')
        x = self.tensor
        def _backward() :
            self.gradient += np.where(x>0,1,0) * out.gradient
        out._backwardpass = _backward
        return out 
    
    def leaky_relu(self,alpha) : 
        @jit(nopython=True,cache=True) 
        def forward (x,alpha) :
            return np.where(x > 0,x,x * alpha) 
        @jit(nopython=True,cache=True)
        def backward_leaky_relu (x,alpha) :
            return np.where(x > 0,1,alpha)
        out = forward(self.tensor,alpha)
        if self.active_grad is True :
            outputs = GradientReflector(out,(self,),'leaky_relu')
        else :
            outputs = GradientReflector(out,_op='leaky_relu')
        def _backward () :
            grad = backward_leaky_relu(self.tensor,alpha)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def tanh (self) :
        x = self.tensor
        @jit(nopython=True,cache=True)
        def count_tanh(x) :
            return np.tanh(x)
        if self.active_grad is True :
            outputs = GradientReflector(count_tanh(x),(self,),'tanh')
        else : 
            outputs = GradientReflector(count_tanh(x),_op='tanh')
        def _backward() :
            self.gradient += (1 - np.tanh(x)**2)* outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def swish(self, Beta=1.0):
        @jit(nopython=True, cache=True)
        def forward(x, beta):
            sigmoid_values = 1 / (1 + np.exp(-beta * x))
            return x * sigmoid_values

        @jit(nopython=True, cache=True)
        def backward(x, beta):
            s = 1 / (1 + np.exp(-beta * x))
            return s + beta * x * s * (1 - s)

        out_tensor = forward(self.tensor, Beta)
        if self.active_grad is True :
            outputs = GradientReflector(out_tensor, (self,), 'swish')
        else :
            outputs = GradientReflector(out_tensor, _op='swish')

        def _backward():
            grad = backward(self.tensor, Beta)
            self.gradient += grad * outputs.gradient

        outputs._backwardpass = _backward
        return outputs
    
    def gelu(self) :
        @jit(nopython=True,cache=True)
        def forward (x) :
            coef = 1.702
            sigmoid_values = 1 / (1 + np.exp(-x * coef))
            return x * sigmoid_values
        
        @jit (nopython=True,cache=True) 
        def backward_gelu (x) :
            coef = 1.702
            sigmoid_values = 1 / (1 + np.exp(-x * coef))
            derivative_s = sigmoid_values * (1 - sigmoid_values)
            grad = sigmoid_values + x * derivative_s * coef 
            return grad 
        if self.active_grad is True :
            outputs = GradientReflector(forward(self.tensor),(self,),'gelu') 
        else :
            outputs = GradientReflector(forward(self.tensor),_op='gelu')
        def _backward () :
            grad  = backward_gelu(self.tensor)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def linear (self) :
        if self.active_grad is True :
            outputs = GradientReflector(self.tensor,(self,),'linear')
        else : 
            outputs = GradientReflector(self.tensor,_op='linear')
        def _backward () :
            self.gradient += np.ones_like(self.tensor) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def softmax(self,axis=-1, keepdims=True, epsilon=1e-6, use_categorical=False):
        try:
            if axis == -1 and keepdims is False:
                raise RuntimeError("if use axis -1 must keepdims = True")
            
            x_max = np.max(self.tensor,axis=axis,keepdims=keepdims)
            x_exp = np.exp(self.tensor - x_max)
            x_sum = np.sum(x_exp,axis=axis,keepdims=keepdims)
            
            @jit(nopython=True, cache=True)
            def div_values(exp_values,sum_values,epsilon):
                sum_values += epsilon
                return exp_values / sum_values
            
            scores = div_values(x_exp,x_sum,epsilon)
            if self.active_grad is True :
                outputs = GradientReflector(scores, (self,), 'softmax')
            else :
                outputs = GradientReflector(scores, _op= 'softmax')
            def _backward():
                if use_categorical == True  :
                    grad = outputs.gradient
                else  :
                    grad_out = outputs.gradient 
                    step1 = np.sum(scores * grad_out,axis=axis,keepdims=keepdims)
                    grad = scores * (grad_out - step1)
                self.gradient += grad

            outputs._backwardpass = _backward
            return outputs

        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            raise

    def exp(self) :
        if self.active_grad is True :

            outputs = GradientReflector((np.exp(self.tensor)),(self,),'exp')
        else : 
            outputs = GradientReflector((np.exp(self.tensor)),_op='exp')
        def _backward() :
            self.gradient += np.exp(self.tensor) * outputs.gradient 
        outputs._backwardpass = _backward
        return outputs
        
    def log(self) :
        if self.active_grad is True :
            outputs = GradientReflector((np.log(self.tensor)),(self,),'log')
        else : 
            outputs = GradientReflector((np.log(self.tensor)),_op='log')
        def _backward() :
            self.gradient += (1 / self.tensor) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def matmul (self,other,transpose_a=False,transpose_b = False) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)

        a_values = self.tensor.swapaxes(-1,-2) if transpose_a else self.tensor
        b_values = other.tensor.swapaxes(-1,-2) if transpose_b else other.tensor
        result = np.matmul(a_values,b_values)
        if self.active_grad is True :
            outputs = GradientReflector(result,(self,other),'matmul')
        else :
            outputs = GradientReflector(result,_op='matmul')
        def _backward () :
            if transpose_a :
                b = b_values.swapaxes(-1,-2)
                a = a_values.swapaxes(-1,-2)
                grad_a = np.matmul(outputs.gradient,b)
                grad_b = np.matmul(a,outputs.gradient)
                grad_a = np.swapaxes(grad_a,-1,-2)
                self.gradient += grad_a
                other.gradient += grad_b 
            elif transpose_b :
                a = a_values.swapaxes(-1,-2)
                b = b_values.swapaxes(-1,-2)
                grad_a = np.matmul(outputs.gradient,b)
                grad_b = np.matmul(a,outputs.gradient)
                self.gradient += grad_a
                other.gradient += np.swapaxes(grad_b,-1,-2)
            else :
                a = self.tensor.swapaxes(-1,-2)
                b = other.tensor.swapaxes(-1,-2)
                self.gradient += np.matmul(outputs.gradient,b)
                other.gradient += np.matmul(a,outputs.gradient)
        outputs._backwardpass = _backward
        return outputs 
    
    def dot (self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        @jit(nopython=True,cache=True)
        def dot_product (x,b):
            return np.dot(x,b)
        if self.active_grad is True :
            outputs = GradientReflector(dot_product(self.tensor,other.tensor),(self,other),'dot')
        else :
            outputs = GradientReflector(dot_product(self.tensor,other.tensor),_op='dot')
        def _backward() :
            a = self.tensor.T 
            b = other.tensor.T
            self.gradient += dot_product(outputs.gradient,b)
            other.gradient += dot_product(a,outputs.gradient)
        outputs._backwardpass = _backward
        return outputs
    
    def sin(self) :
        if self.active_grad is True :
            outputs = GradientReflector(np.sin(self.tensor),(self,),'sin')
        else :
            outputs = GradientReflector(np.sin(self.tensor),_op='sin')
        def _backward() :
            self.gradient += np.cos(self.tensor) * outputs.gradient
        
        outputs._backwardpass = _backward
        return outputs 
    
    def cos (self) :
        if self.active_grad is True :
            outputs = GradientReflector(np.cos(self.tensor),(self,),'cos')
        else :
            outputs = GradientReflector(np.cos(self.tensor),_op='cos')
        def _backward():
            self.gradient += -np.sin(self.tensor) * outputs.gradient

        outputs._backwardpass = _backward
        return outputs
    
    def tan(self) :
        if self.active_grad is True :

            outputs = GradientReflector(np.tan(self.tensor),(self,),'tan')
        else :
            outputs = GradientReflector(np.tan(self.tensor),_op='tan')
        def _backward() :
            self.gradient += (1 / np.power(np.cos(self.tensor),2)) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def clip (self,min_vals,max_vals) :
        if self.active_grad is True :
            outputs = GradientReflector(np.clip(self.tensor,min_vals,max_vals),(self,),'clip')
        else : 
            outputs = GradientReflector(np.clip(self.tensor,min_vals,max_vals),_op='clip')
        def _backward() :
            grad = (self.tensor >= min_vals) & (self.tensor <= max_vals)
            self.gradient += grad.astype(float) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def __getitem__(self,idx) :
        if self.active_grad is True:

            outputs = GradientReflector(self.tensor[idx],(self,),f'getitem{idx}')
        else : 
            outputs = GradientReflector(self.tensor[idx],_op='getitem{idx}')
        def _backward () :
            grad = np.zeros_like(self.tensor)
            grad[idx] = outputs.gradient 
            self.gradient += grad
        outputs._backwardpass = _backward
        return outputs 
    
    def reshape(self,shape=()) :
        if self.active_grad :
            outputs = GradientReflector(np.reshape(self.tensor,shape),(self,),'reshape')
        else : 
            outputs = GradientReflector(np.reshape(self.tensor,shape),_op='reshape')
        def _backward () :
            self.gradient += np.reshape(outputs.tensor,(self.tensor.shape))
        outputs._backwardpass = _backward
        return outputs 
    
    def sigmoid(self) :
        @jit(cache=True,nopython=True)
        def forward(x) :
            return 1 / (1 + np.exp(-x))
        
        @jit(cache=True,nopython=True)
        def backward_sigmoid (s) :
            return s * (1 - s)
        
        result = forward(self.tensor)
        if self.active_grad is True :
            outputs = GradientReflector(result,(self,),'sigmoid')
        else :
            outputs = GradientReflector(result,_op='sigmoid')
        def _backward () :
            s = outputs.tensor
            grad = backward_sigmoid(s)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    

    def binarycrossetnropy (self,y_true,epsilon=1e-6,from_logits = False) :
        """
            how to use
            -----------
            y_pred = its model predicted data, as usually its always Tensor for Engine grad\n 
            y_true = the true of label datasets \n 
            y_pred.binarycrossetnropy(y_true)

            Author
            ----------
            Candra Alpin Gunawan 
        """
        if isinstance(y_true,GradientReflector) : 
            n = len(y_true.tensor)
            y = y_true.tensor 
        else : 
            n = len(y_true) 
            y = y_true 
        y_pred = self.tensor

        if from_logits is True :
            y_pred = 1 / (1 + np.exp(-y_pred))

        y_pred = np.clip(y_pred,epsilon,1-epsilon)
        loss = (-1/n) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        if self.active_grad is True :
            outputs = GradientReflector(loss,(self,),'binary_crossentropy')
        else :
            outputs = GradientReflector(loss,_op='binary_crossentropy')

        def _backward () :
            grad = (y_pred - y_true) / n               
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def categoricallcrossentropy (self,y_true,epsilon=1e-6,from_logits=False) :
        """
            how to use
            -----------
            y_pred = its model predicted data, as usually its always Tensor for Engine grad\n 
            y_true = the true of label datasets \n 
            y_pred.categoricallcrossentropy(y_true)

            Author
            ----------
            Candra Alpin Gunawan 
        """
        if isinstance(y_true,GradientReflector) : 
            y = y_true.tensor
        else : 
            y = y_true

        y_pred = self.tensor
        if from_logits is True :
            x_max = np.max(y_pred,axis=-1,keepdims=True)
            x_exp = np.exp(y_pred - x_max)
            x_sum = np.sum(x_exp,axis=-1,keepdims=True)
            x_sum[x_sum==0] = epsilon
            y_pred = x_exp / x_sum 
        y_pred = np.clip(y_pred,epsilon,1-epsilon)
        if y_pred.ndim <= 2 :
            loss = -np.sum((y) * np.log(y_pred))
            loss = np.mean(loss)
        else : 
            loss = list()
            for i in range(y_pred.shape[0]) :
                loss_values = -np.sum(y[i] * np.log(y_pred[i]))
                loss.append(loss_values)
            loss = np.array(loss).mean()
        if self.active_grad is True :
            outputs = GradientReflector(loss,(self,),'categoricall_crossentropy')
        else :
            outputs = GradientReflector(loss,_op='categoricall_crossentropy')
        def _backward () :
            if y_pred.ndim <= 2:
                grad = (y_pred - y) / y.shape[0]
            else :
                grad = list()
                for i in range(y_pred.shape[0]) :
                    grad.append((y_pred[i] - y[i])/len(y[i]))
                grad = np.array(grad).sum()
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def sparsecategoricallcrossentropy (self,y_true,epsilon=1e-6,from_logits=False) :
        """
            how to use
            -----------
            y_pred = its model predicted data, as usually its always Tensor for Engine grad\n 
            y_true = the true of label datasets \n 
            y_pred.sparsecategoricallcrossentropy(y_true)

            Author
            ----------
            Candra Alpin Gunawan 
        """
        if isinstance(y_true,GradientReflector) : 
            y = y_true.tensor 
        else :
            y = y_true 

        y_pred = self.tensor
        if from_logits is True  :
            x_max = np.max(y_pred,axis=-1,keepdims=True)
            x_exp = np.exp(y_pred - x_max)
            x_sum = np.sum(x_exp,axis=-1,keepdims=True)
            x_sum[x_sum==0] = epsilon 
            y_pred = x_exp / x_sum
        if y_pred.ndim <=2 :
            y_pred = np.clip(y_pred,epsilon,1 - epsilon)
            labels_indeks_True = np.arange(len(y))
            Loss = -np.log(y_pred[labels_indeks_True,y])
            out = np.mean(Loss)
        else :
            y_pred = np.clip(y_pred,epsilon,1-epsilon)
            cache = list()
            def count_loss (y_true,y_pred) :
                labels_indeks_True = np.arange(len(y_true))
                loss = -np.log(y_pred[labels_indeks_True,y_true])
                return np.mean(loss),labels_indeks_True
            out = list()
            for i in range(y.shape[0]) :
                tensor_out,cc = count_loss(y[i],y_pred[i])
                out.append(tensor_out)
                cache.append(cc)
            out = np.array(out).mean()
        
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,),'sparse_categoricallcrossentropy')
        else :
            outputs = GradientReflector(out,_op='sparse_categoricallcrossentropy')

        def _backward() :
            if y_pred.ndim <=2:
                grad = y_pred.copy()
                grad[labels_indeks_True,y] -= 1
                grad_ = grad / len(y)
                self.gradient += grad_ * outputs.gradient
            else :
                pred_tensor = y_pred.copy()
                grad_n = np.zeros_like(pred_tensor)
                for i in range(pred_tensor.shape[0]) :
                    a = pred_tensor[i]
                    a[cache[i],y[i]] -=1 
                    grad_n += a / len(y[i])
                self.gradient += grad_n * outputs.gradient 
        outputs._backwardpass = _backward
        return outputs 
    
    def meansquareerror (self,y_true) :
        """
            how to use
            -----------
            y_pred = its model predicted data, as usually its always Tensor for Engine grad\n 
            y_true = the true of label datasets \n 
            y_pred.meansquareerror(y_true)

            Author
            ----------
            Candra Alpin Gunawan 
        """
        if isinstance(y_true,GradientReflector) : 
            y = y_true.tensor 
        else : 
            y = y_true
        y_pred = self.tensor
        
        if y_pred.ndim > 2 :
            loss = 0 
            for i in range(y_pred.shape[0]) :
                loss += np.mean(np.power((y_pred[i] - y[i]),2))
            loss = np.mean(loss)
        else :
            loss = np.mean(np.power((y_pred - y),2))
        if self.active_grad is True:
            outputs = GradientReflector(loss,(self,),'mse')
        else :
            outputs = GradientReflector(loss,'mse')

        def _backward () :
            if y_pred.ndim <=2 :
                grad = (2/len(y)) * (y_pred - y)
                self.gradient += grad * outputs.gradient
            else :
                grad = np.zeros_like(self.gradient)
                for i in range(y_pred.shape[0]) :
                    grad += (2/len(y[i])) * (y_pred[i] - y[i])
                self.gradient += grad * outputs.gradient

        outputs._backwardpass = _backward
        return outputs

    def meanabsoluteerror (self,y_true) :
        """
            how to use
            -----------
            y_pred = its model predicted data, as usually its always Tensor for Engine grad\n 
            y_true = the true of label datasets \n 
            y_pred.meanabsoluteerror(y_true)

            Author
            ----------
            Candra Alpin Gunawan 
        """
        if isinstance(y_true,GradientReflector) : 
            y = y_true.tensor 
        else : 
            y = y_true 
        y_pred = self.tensor
        if y_pred.ndim > 2 :
            loss = 0 
            for i in range(y_pred.shape[0]) :
                loss += np.mean(np.abs(y_pred[i] - y[i]))
            loss = np.mean(loss)
        else :
            loss = np.mean(np.abs(y_pred - y))
        if self.active_grad is True:
            outputs = GradientReflector(loss,(self,),'mae')
        else :
            outputs = GradientReflector(loss,_op='mae')
        def _backward() :
            if y_pred.ndim <= 2 :    
                grad =  (1/(len(y))) * np.sign(y_pred - y)
                self.gradient += grad * outputs.gradient
            else :
                grad = np.zeros_like(self.gradient)
                for i in range(y_pred.shape[0]) :
                    grad += (1/(len(y[i]))) * np.sign(y_pred[i] - y[i])
                self.gradient += grad * outputs.gradient

        outputs._backwardpass = _backward
        return outputs
    
    def hubber_loss (self,y_true,delta=1.0) : 
        """
            how to use
            -----------
            y_pred = its model predicted data, as usually its always Tensor for Engine grad\n 
            y_true = the true of label datasets \n 
            y_pred.hubber_loss(y_true)

            Author
            ----------
            Candra Alpin Gunawan 
        """
        if isinstance(y_true,GradientReflector) : 
            y = y_true.tensor 
        else : 
            y = y_true 
        y_pred = self.tensor
        if y_pred.ndim > 2 :
            loss = 0 
            for i in range(y_pred.shape[0]) : 
                lt = y_pred[i] - y[i]
                hb_t = np.where(np.abs(lt)<= delta, 0.5 * np.mean(np.power(lt,2)),
                                    delta * (np.abs(lt) - (0.5 * delta)))
                loss +=hb_t
            hub_loss = np.mean(loss)
        else:
            loss = y_pred - y
            hub_loss = np.where(np.abs(loss) <= delta,0.5 * np.mean(np.power(loss,2)),
                                delta * (np.abs(loss) - (0.5 * delta)))
            hub_loss = np.mean(hub_loss)
        if self.active_grad is True :
            outputs = GradientReflector(hub_loss,(self,),'huber_loss')
        else :
            outputs = GradientReflector(hub_loss,_op='huber_loss')
        def _backward() :
            if y_pred.ndim > 2 :
                grad = np.zeros_like(self.gradient)
                for i in range(y_pred.shape[0]) :
                    loss = y_pred[i] - y[i]
                    grad += np.where(np.abs(loss) <= delta,loss,delta * np.sign(loss))
                self.gradient += grad * outputs.gradient
            else :
                grad = np.where(np.abs(loss) <= delta,loss,delta * np.sign(loss))
                self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 

    def sqrt(self) :
        x = self.tensor 
        if self.active_grad is True :
            outputs = GradientReflector(np.sqrt(x),(self,),_op='sqrt')
        else :
            outputs = GradientReflector(np.sqrt(x),(self,),_op='sqrt')
        def backward() : 
            grad = 1 / (2 * np.sqrt(x))
            self.gradient += grad * outputs.gradient
        
        outputs._backwardpass = backward 
        return outputs

    def sum(self,axis=None,keepdims=False) :
        values = np.sum(self.tensor,axis=axis,keepdims=keepdims)
        if self.active_grad is True :
            outputs = GradientReflector(values,(self,),'sum')
        else : 
            outputs = GradientReflector(values,_op='sum')
        def _backward () :
            self.gradient += np.ones_like(self.tensor) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def transpose(self,shape=()) :
        shape = shape 
        tensor = self.tensor
        if self.active_grad is True :
            outputs = GradientReflector(np.transpose(tensor,axes=shape),(self,),'transpose')
        else :
            outputs = GradientReflector(np.transpose(tensor,axes=shape),_op='transpose')
        def _backward () :
            self.gradient = np.transpose(outputs.gradient,shape)
        outputs._backwardpass = _backward
        return outputs
    
    def max(self,axis=None,keepdims=False) :
        out = np.max(self.tensor,axis=axis,keepdims=keepdims)
        if self.active_grad is True :
            outputs = GradientReflector(out,(self,),'max')
        else : 
            outputs = GradientReflector(out,_op='max')
        def _backward() :
            masked = np.equal(self.tensor, np.max(self.tensor, axis=axis, keepdims=True)).astype(float)

            grad = outputs.gradient 
            if axis is not None :
                grad = np.expand_dims(grad,axis=axis)
            maksed_grad = masked * grad
            self.gradient = maksed_grad
        outputs._backwardpass = _backward
        return outputs 
    
    def variace(self,axis=None,keepdims=False) :
        
        x = self.tensor 
        means = np.mean(x,axis=axis,keepdims=keepdims) 
        var = np.mean(np.power((x - means),2),axis=axis,keepdims=keepdims)
        if self.active_grad is True:
            outputs = GradientReflector(var,(self,),'variance')
        else :
            outputs = GradientReflector(var,_op='variance')
        def _backward() :
            N = x.size if axis is None else x.shape[axis]
            grad = (2 / N) * (x - means)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def std(self,axis=None,keepdims=False,epsilon=1e-7) :
        x = self.tensor
        mean = np.mean(x,axis=axis,keepdims=keepdims)
        var = np.mean((x - mean)**2,axis=axis,keepdims=keepdims)
        std_vals = np.sqrt(var + epsilon)
        if self.active_grad is True:
            outputs = GradientReflector(std_vals,(self,),'std')
        else :
            outputs = GradientReflector(std_vals,_op='std')
        def _backward() :
            N = np.prod(x.shape) if axis is None else x.shape[axis]
            broadcast_mean = mean if axis is None else np.expand_dims(mean,axis=axis)
            broadcast_std = std_vals if axis is None else np.expand_dims(std_vals,axis=axis)

            grad = (x - broadcast_mean) / (N * broadcast_std)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs

    def layernormalization_backend(self, gamma, beta, epsilon=1e-6):
        """
            how to use it
            --------------
                x = GradientReflector([[2,3,4,2][9,2,3,4]])\n
                gamma = np.ones((1,x.shape[-1]))\n
                beta = np.zeros((1,x.shape))\n
                x.layernormalization_backend(gamma,beta)
            
            Author
            ---------
             Candra Alpin Gunawan
        """
        gamma = gamma if isinstance(gamma, GradientReflector) else GradientReflector(gamma)
        beta = beta if isinstance(beta, GradientReflector) else GradientReflector(beta)
        x = self.tensor
        mean = np.mean(x,axis=-1,keepdims=True)
        var = np.mean((x - mean)**2,axis=-1,keepdims=True)
        x_mu = x - mean 
        std = np.sqrt(var + epsilon )
        @jit(nopython=True,cache=True) 
        def count_forward(x_mu,std,gamma,beta) :
            x_hat = x_mu / std
            return gamma * x_hat +  beta 
          
        @jit(nopython=True,cache=True)
        def count_backward(dx_hat,std,dvar,x_mu,N,d_mean) :
            return dx_hat / std + dvar * 2.0 * x_mu / N + d_mean / N
            
        out = count_forward(x_mu=x_mu,std=std,gamma=gamma.tensor,beta= beta.tensor)
        if self.active_grad is True:
            output = GradientReflector(out, (self, gamma, beta), 'layer_normal_backend')
        else :
            output = GradientReflector(out,_op= 'layer_normal_backend')

        def _backward():
            N = x.shape[-1]
            x_hat = x_mu / std 
            dy = output.gradient
            axes = tuple(range(output.gradient.ndim - 1))
            beta.gradient += np.sum(dy,axis=axes)
            gamma.gradient += np.sum((dy * x_hat),axis=(0,1))
            dx_hat = dy * gamma.tensor
            d_var = np.sum(dx_hat * x_mu * -0.5 * std**-3,axis=-1,keepdims=True)
            dmean = np.sum(dx_hat * -1.0 / std,axis=-1,keepdims=True) - d_var * np.mean(-2.0 * x_mu,axis=-1,keepdims=True)
            grad = count_backward(dx_hat,std,d_var,x_mu,N,dmean)
            self.gradient += grad 

        output._backwardpass = _backward
        return output

    def bacthnormalization_backend(self,gamma,beta,epsilon=1e-6) :
        """
            how to use it
            --------------
                x = GradientReflector([[2,3,4,2][9,2,3,4]])\n
                gamma = np.ones((1,x.shape[-1]))\n
                beta = np.zeros((1,x.shape))\n
                x.bacthnormalization_backend(gamma,beta)
            
            Author
            ---------
             Candra Alpin Gunawan
        """
        gamma = gamma if isinstance(gamma,GradientReflector) else GradientReflector(gamma)
        beta = beta if isinstance(beta,GradientReflector) else GradientReflector(beta)
        x = self.tensor
        mean = np.mean(self.tensor,axis=0,keepdims=True)
        var = np.mean(np.power((self.tensor - mean),2),axis=0,keepdims=True)
        x_mu = x - mean
        std = np.sqrt(var + epsilon) 
        normal = x_mu / std 
        
        @jit(nopython=True,cache=True)
        def count_forward(x_mu,std,gamma,beta) :
            x_hat = x_mu / std 
            return gamma * x_hat + beta 
        
        @jit(nopython = True , cache=True)
        def count_backward (d_normal,std_inv,dvar,x_mu,N,dmean) :
            return d_normal * std_inv + dvar * 2.0 * (x_mu/N) + (dmean/N)

        out = count_forward(x_mu= x_mu,std=std,gamma=gamma.tensor,beta=beta.tensor)
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,gamma,beta),'batch_normal_backend')
        else :
            outputs = GradientReflector(out,_op='batch_normal_backend')

        def _backward() :
            N = self.tensor.shape[0]
            x_mu = self.tensor - mean
            std_inv = 1 / np.sqrt(var + epsilon)
            axes = tuple(range(outputs.gradient.ndim - 1))
            beta.gradient += np.sum(outputs.gradient,axis=axes)
            gamma.gradient += np.sum((outputs.gradient * normal),axis=axes)
            d_normal = outputs.gradient * gamma.tensor
            dvar = np.sum((d_normal * x_mu * -0.5 * std_inv**3),axis=0,keepdims=True)
            dmean = np.sum((d_normal * -std_inv),axis=0,keepdims=True) + dvar * np.mean((-2 * x_mu),axis=0,keepdims=True)
            grad = count_backward(d_normal = d_normal,std_inv=std_inv,dvar=dvar,x_mu=x_mu,N=N,dmean=dmean)
            self.gradient += grad 
        outputs._backwardpass = _backward
        return outputs

    def get_tensor(self) :
        return self.tensor
    
    def singlehead_attention_backend (self,Keys,Values,weight=(),bias=(),use_causal_mask=False,stop_jacobian=True,
                                      softmax_derivative : Literal ['Alternative','Jacobian'] = 'Alternative'):
        """
            singlehead_attention_backendd its backend for Attention Layers. singlehead_attention_backend working for 
            forward and backward Attention layers. \n 
            MultiHeadAttention working by 3 weight following 3 bias too, weight query,weight key,weight values 
            and from bias is bias query , bias keys, bias values. 

            this back end layers have two softmax derivative mode that is jacobian and Alternative, in default Jacobian not used 
            cause memorie and stable problem, but if for researching gradient with super high computer its can use too. 

            how to use: 

            import numpy as np \n 
            from LittleLearn import GradientReflector 

            query = np.random.rand(10,15,32)
            
            keys = np.random.rand(10,15,32)

            values = np.random.rand(10,15,32)

            features = 32 \n 
            units = 16 \n 

            w_query = np.random.rand(features,units)\n
            w_keys = np.random.rand(features,units)\n
            w_values = np.random.rand(features,units)\n

            b_query = np.zeros((1,units))\n 
            b_keys = np.zeros((1,units))\n 
            b_values = np.zeros((1,units))\n 

            query = GradientReflector(query)

            keys = GradientReflector(keys)

            values = GradientReflector(values)

            query.singlehead_attention_backend(keys,values,
            weight=(w_query,w_keys,w_values,w_output),bias=(b_query,b_keys,b_values,b_output),
            use_causal_mask = False, stop_jacobian = True, softmax_derivative = 'Alternative')

            use_causal_mask False for default 

            stop_jacobian True for default 

            softmax_derivatibe Alternative for default 

            if want use_causal_mask for decoder Transformers and need to close next token asumtion 
            you can set use_causal_mask to be True, if you in researching for new architecture or about gradient 
            in Attention layers you can set stop_jacobian = False and softmax_derivative = 'Jabobian'

            
        """

        if len(weight) != 3 :
            raise RuntimeError("Weight must for Query,Keys,Values")
        if len(bias) != 3 :
            raise RuntimeError("Bias must for Query,Keys,Values")
        wq,wk,wv = weight
        bq,bk,bv = bias
        wq = wq if isinstance(wq,GradientReflector) else GradientReflector(wq,_op='WeightQuery')
        wk = wk if isinstance(wk,GradientReflector) else GradientReflector(wk,_op='WeightKeys')
        wv = wv if isinstance(wv,GradientReflector) else GradientReflector(wv,_op='WeightValues')
        bq = bq if isinstance(bq,GradientReflector) else GradientReflector(bq,_op='biasQuery')
        bk = bk if isinstance(bk,GradientReflector) else GradientReflector(bk,_op='BiasKeys')
        bv = bv if isinstance(bv,GradientReflector) else GradientReflector(bv,_op='BiasValues')
        Query = self.tensor
        Keys = Keys if isinstance(Keys,GradientReflector) else GradientReflector(Keys)
        Values = Values if isinstance(Values,GradientReflector) else GradientReflector(Values)
        @jit(nopython=True,cache=True)
        def alternative_counting (step1,softmax_score,outgrad) :
            return softmax_score * (outgrad - step1)
        
        @jit(nopython=True,cache=True)
        def softmax_counter (x_exp,x_sum) :
            x_sum = np.where(x_sum <=0,1e-8, 1 - 1e-8)
            return x_exp / x_sum

        def softmax(self,x) :
            x_max = np.max(x,axis=-1,keepdims=True)
            x_exp = np.exp(x - x_max)
            x_sum = np.sum(x_exp,axis=-1,keepdims=True)
            result = softmax_counter(x_exp,x_sum)
            return result 
        
        def jacobian_derivative (softmax_score,stop_jacobian=True) :
            stop_jacobian = stop_jacobian
            print("Critical Warning : Jacobian Mode is dangerous , you computer may be crash. ")
            if stop_jacobian is True :
                raise RuntimeError("The jacobian mode be stop cause user decision")
            batch,seq,dim = softmax_score
            jacobian = np.zeros((batch,seq,dim,dim),dtype=np.float32)
            for b in range(batch) :
                for s in range(seq) :
                    y = np.reshape(softmax_score[b,s],(-1,1))
                    y = np.diagflat(y) - np.dot(y,y.T)
                    jacobian[b,s] = y 
            return jacobian
        
        def derivative_alternative (x,out_grad) :
            S = x
            grad_out = out_grad
            step1 = np.sum(grad_out * S ,axis=-1,keepdims=True)
            result = alternative_counting(step1,S,out_grad)
            return result
        
        def scale_dot_product (Q,K,V,causal_mask=False) :
            key_dim = K.shape[-1]
            score = np.matmul(Q,np.transpose(K,axes=(0,2,1))) / np.sqrt(key_dim)
            mask = None 
            if causal_mask is True:
                mask = 1 - np.tril(np.ones(shape=(Q.shape[1],Q.shape[1]),dtype=np.float32))
                mask = np.expand_dims(mask,axis=0)
                score += (mask * -1e9)
            score = softmax(score)
            attention = np.matmul(score,V)
            return attention,score,mask
        
        Q = np.matmul(Query,wq.tensor) + bq.tensor 
        K = np.matmul(Keys.tensor,wk.tensor) + bk.tensor 
        V = np.matmul(Values.tensor,wv.tensor) + bv.tensor 
        attention,score,mask = scale_dot_product(Q,K,V,causal_mask=use_causal_mask)
        if self.active_grad is True :
            outputs = GradientReflector(attention,(self,Keys,Values,wq,wk,wv,bq,bk,bv),"attention")
        else :
            outputs = GradientReflector(attention,_op="attention")

        def _backward() :
            key_dim = K.shape[-1]
            grad_out = outputs.gradient
            grad_Wv = np.matmul(score.transpose(0,2,1),grad_out)
            d_score = np.matmul(grad_out,V.transpose(0,2,1))
            if softmax_derivative == 'Jacobian' :
                d_softmax = jacobian_derivative(softmax_score=score,stop_jacobian=stop_jacobian)
                d_softmax = np.einsum("bsij,bsj->bsi",d_softmax,d_score)
            elif softmax_derivative == 'Alternative':
                d_softmax = derivative_alternative(score,d_score)
            else :
                raise RuntimeError("Mode just available for Jacobian and Alternative")         
            if use_causal_mask is True :
                d_softmax *= (1 - mask)
            d_q = np.matmul(d_softmax,K) / np.sqrt(key_dim)
            d_k = np.matmul(d_softmax,Q) / np.sqrt(key_dim)
            grad_q = np.matmul(Query.transpose(0,2,1),d_q)
            grad_k = np.matmul(Keys.transpose(0,2,1),d_k)
            grad_v = np.matmul(Values.transpose(0,2,1),grad_Wv)
            grad_bq = np.sum(grad_q,axis=1)
            grad_bk = np.sum(grad_k,axis=1)
            grad_bv = np.sum(grad_v,axis=1)
            grad_bq = np.mean(grad_bq,axis=0,keepdims=True)
            grad_bk = np.mean(grad_bk,axis=0,keepdims=True)
            grad_bv = np.mean(grad_bv,axis=0,keepdims=True)
            grad_nq = np.matmul(grad_q,wq.tensor.T)
            grad_nk = np.matmul(grad_k,wk.tensor.T)
            grad_nv = np.matmul(grad_v,wv.tensor.T)
            wq.gradient += grad_q
            wk.gradient += grad_k
            wv.gradient += grad_v
            bq.gradient += grad_bq
            bk.gradient += grad_bk
            bv.gradient += grad_bv
            self.gradient += grad_nq
            Keys.gradient += grad_nk
            Values.gradient += grad_nv
        outputs._backwardpass = _backward
        return outputs

    def multihead_attention_backend(self,Keys,Values,num_head,weight=(),bias=(),use_causal_mask=False,
                                    stop_jacobian=True,
                                    softmax_derivative : Literal['Jacobian','Alternative'] = 'Alternative'):
        """
            multihead_attention_backend its backend for MultiHeadAttention Layers. multihead_attention_backend working for 
            forward and backward MultiHeadAttention layers. \n 
            MultiHeadAttention working by 4 weight following 4 bias to, weight query,weight key,weight values, weight output 
            and from bias is bias query , bias keys, bias values, bias output. 

            this back end layers have two softmax derivative mode that is jacobian and Alternative, in default Jacobian not used 
            cause memorie and stable problem, but if for researching gradient with super high computer its can use too. 

            how to use: 

            import numpy as np \n 
            from LittleLearn import GradientReflector 

            query = np.random.rand(10,15,32)
            
            keys = np.random.rand(10,15,32)

            values = np.random.rand(10,15,32)

            features = 32 \n 
            units = 16 \n 

            w_query = np.random.rand(features,units)\n
            w_keys = np.random.rand(features,units)\n
            w_values = np.random.rand(features,units)\n
            w_output = np.random.rand(units,units)\n

            b_query = np.zeros((1,units))\n 
            b_keys = np.zeros((1,units))\n 
            b_values = np.zeros((1,units))\n 
            b_output  = np.zeros((1,units))\n 

            query = GradientReflector(query)

            keys = GradientReflector(keys)

            values = GradientReflector(values)

            query.multihead_attention_backend(keys,values,
            weight=(w_query,w_keys,w_values,w_output),bias=(b_query,b_keys,b_values,b_output),
            use_causal_mask = False, stop_jacobian = True, softmax_derivative = 'Alternative')

            use_causal_mask False for default 

            stop_jacobian True for default 

            softmax_derivatibe Alternative for default 

            if want use_causal_mask for decoder Transformers and need to close next token asumtion 
            you can set use_causal_mask to be True, if you in researching for new architecture or about gradient 
            in MultiHeadAttention layers you can set stop_jacobian = False and softmax_derivative = 'Jabobian'

            Reference:
            ----------
            Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
            https://arxiv.org/abs/1706.03762

            Written by: Candra Alpin Gunawan
        """
        if len(weight) !=4 :
            raise RuntimeError("Weight must for Query,Keys,Values,Output")
        if len(bias) !=4 :
            raise RuntimeError("Bias must for Query,Keys,Values,Output")
        wq,wk,wv,wo = weight
        bq,bk,bv,bo = bias 
        Query = self.tensor
        Keys = Keys if isinstance(Keys,GradientReflector) else GradientReflector(Keys,_op="Keys")
        Values = Values if isinstance(Values,GradientReflector) else GradientReflector(Values,_op="Values")
        wq = wq if isinstance(wq,GradientReflector) else GradientReflector(wq,_op='WeightQuery')
        wk = wk if isinstance(wk,GradientReflector) else GradientReflector(wk,_op='WeightKeys')
        wv = wv if isinstance(wv,GradientReflector) else GradientReflector(wv,_op='WeightValues')
        wo = wo if isinstance(wo,GradientReflector) else GradientReflector(wo,_op='WeightOutput')
        bq = bq if isinstance(bq,GradientReflector) else GradientReflector(bq,_op='BiasQuery')
        bk = bk if isinstance(bk,GradientReflector) else GradientReflector(bk,_op='BiasKeys')
        bv = bv if isinstance(bv,GradientReflector) else GradientReflector(bv,_op='BiasValues')
        bo = bo if isinstance(bo,GradientReflector) else GradientReflector(bo,_op='BiasOutput')
        
        def softmax(x) :
            x_max = np.max(x,axis=-1,keepdims=True)
            x_exp = np.exp(x - x_max)
            x_sum = np.sum(x_exp,axis=-1,keepdims=True)
            x_sum[x_sum==0] = 1e-8
            result = x_exp / x_sum 
            return result 
        
        def Jacobian_mode (score,stop_jacobian=True) : 
            stop_jacobian = stop_jacobian
            batch,head,seq,dim = score.shape 
            print("Critical Warning : Jacobian Mode is dangerous , you computer may be crash. ")
            if stop_jacobian is True :
                raise RuntimeError("The jacobian mode be stop cause user decision")
            jacobian = np.zeros((batch,head,seq,dim,dim),dtype=np.float32)
            for b in range(batch) :
                for h in range(head) :
                    for s in range(seq) :
                        y = score[b,h,s].reshape(-1,1)
                        jacob = np.diagflat(y) - np.dot(y,y.T)
                        jacobian[b,h,s] +=jacob
            return jacobian
        
        def Alternative_mode (score,grad_out) :
            step = np.sum(score * grad_out,axis=-1,keepdims=True)
            step2 = score * (grad_out - step)
            return step2 
        
        def splithead (x) :
            batch,seq,dim = x.shape 
            x = np.reshape(x,newshape=(batch,seq,num_head,(dim//num_head)))
            x = np.transpose(x,axes=(0,2,1,3))
            return x 
        
        def scaled_dot_product(Q,K,V,causal_mask = False) : 
            key_dim = K.shape[-1]
            score = np.matmul(Q,np.transpose(K,axes=(0,1,3,2))) / np.sqrt(key_dim)
            mask = None 
            if causal_mask is True :
                mask = 1 - np.tril(np.ones(shape=(Q.shape[2],Q.shape[2]),dtype=np.float32))
                mask = mask[np.newaxis,np.newaxis,:,:]
                score += (mask * -1e9)
            score = softmax(score)
            attention = np.matmul(score,V)
            return attention,score,mask
        
        Q = np.matmul(Query,wq.tensor) + bq.tensor
        K = np.matmul(Keys.tensor,wk.tensor) + bk.tensor
        V = np.matmul(Values.tensor,wv.tensor) + bv.tensor
        batch,seq,dim = Q.shape
        key_dim = dim // num_head
        Q = splithead(Q)
        K = splithead(K)
        V = splithead(V)
        attention,score,mask = scaled_dot_product(Q,K,V,causal_mask=use_causal_mask)
        attention = np.transpose(attention,axes=(0,2,1,3))
        attention = np.reshape(attention,(batch,seq,dim))
        out = np.matmul(attention,wo.tensor) + bo.tensor
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,wq,wk,wv,wo,bq,bk,bv,bo),_op="MultiHeadAttention")
        else :
            outputs = GradientReflector(out,_op="MultiHeadAttention")
        
        def _backward() :
            grad_out = outputs.gradient
            d_Wo = np.matmul(np.transpose(attention,axes=(0,2,1)),grad_out)
            d_attn_fwo = np.matmul(grad_out,wo.tensor.T)
            d_attn_fwo = splithead(d_attn_fwo)
            d_wv = np.matmul(score.transpose(0,1,3,2),d_attn_fwo)
            d_score = np.matmul(d_attn_fwo,V.transpose(0,1,3,2))
            if softmax_derivative == 'Jacobian'  :
                d_softmax = Jacobian_mode(score,stop_jacobian=stop_jacobian)
                d_softmax = np.einsum("bhsdj,bsdj ->bhsd",d_softmax,d_score)
            elif softmax_derivative == 'Alternative' :
                d_softmax = Alternative_mode(score,d_score)
            if use_causal_mask is True :
                d_softmax *= (1 - mask)
            grad_dq = np.matmul(d_softmax,K) / np.sqrt(key_dim)
            grad_dk = np.matmul(d_softmax,Q) / np.sqrt(key_dim)
            grad_dq = np.transpose(grad_dq,axes=(0,2,1,3))
            grad_dq = np.reshape(grad_dq,newshape=(batch,seq,dim))
            grad_dk = np.transpose(grad_dk,axes=(0,2,1,3))
            grad_dk = np.reshape(grad_dk,newshape=(batch,seq,dim))
            grad_dv = np.transpose(d_wv,axes=(0,2,1,3))
            grad_dv = np.reshape(grad_dv,newshape=(batch,seq,dim))
            grad_Wq = np.matmul(Query.transpose(0,2,1),grad_dq)
            grad_Wk = np.matmul(Keys.tensor.transpose(0,2,1),grad_dk)
            grad_Wv = np.matmul(Values.tensor.transpose(0,2,1),grad_dv)
            grad_Bq = np.sum(grad_dq,axis=1)
            grad_Bk = np.sum(grad_dk,axis=1)
            grad_Bv = np.sum(grad_dv,axis=1)
            grad_Bo = np.sum(d_Wo,axis=1)
            grad_Bq = np.mean(grad_Bq,axis=0,keepdims=True)
            grad_Bk = np.mean(grad_Bk,axis=0,keepdims=True)
            grad_Bv = np.mean(grad_Bv,axis=0,keepdims=True)
            grad_Bo = np.mean(grad_Bo,axis=0,keepdims=True)
            self.gradient = np.matmul(grad_dq,wq.tensor.T)
            Keys.gradient = np.matmul(grad_dk,wk.tensor.T)
            Values.gradient = np.matmul(grad_dv,wv.tensor.T)
            wq.gradient += grad_Wq.mean(axis=0)
            wk.gradient += grad_Wk.mean(axis=0)
            wv.gradient += grad_Wv.mean(axis=0)
            wo.gradient += d_Wo.mean(axis=0)
            bq.gradient += grad_Bq
            bk.gradient += grad_Bk
            bv.gradient += grad_Bv
            bo.gradient += grad_Bo

            
        outputs._backwardpass = _backward
        return outputs

    def global_average_pooling_backend (self,axis=0,keepdims=False) :
        """
            global_average_pooling_backend is GlobalAveragePooling layers back end, it working for forward and bacward 
            layers. \n 
            GlobalAveragePooling layers has not have layers. its just for Average logits that from the some layers 
            give big dimention output but the next layers just suport more low dimention input. 

            how to use:
            
            import numpy as np \n 
            from LittleLearn import GradientReflector \n 

            x = np.random.rand(10,15,32) 
            
            x = GradientReflector(x)

            x.global_average_pooling_backend(axis=0,keepdims=False)

            axis default by 0 

            keepdims default by False 

            if want make dimention not change cause this operation, set keepdims tu be True 
        
        """
        x = self.tensor 
        out = np.mean(x,axis=axis,keepdims=keepdims)
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,),'globalaveragepooling')
        else :
            outputs = GradientReflector(out,_op='globalaveragepooling')
        def _backward() :
            if axis is None :
                N = x.size 
            elif isinstance(axis,int) :
                N = x.shape[axis]
            else :
                N =1 
                for ax in axis :
                    N *= x.shape[ax]
             
            grad = outputs.gradient
            
            if keepdims is False :
                grad = np.expand_dims(grad,axis=axis)
            gradient = np.broadcast_to(grad/N,x.shape)
            
            self.gradient+= gradient
        
        outputs._backwardpass = _backward
        return outputs 

    def simple_rnn_backend (self,hidden_state,weight=(),bias_=None) : 
        """
            simple_rnn_backend is backend at SimpleRNN layers. its working for handle forward backward 
            SimpleRNN layers. \n 
            SimpleRNN layers work by 2 weight bu just 1 bias. weight is for sequence and hidden state 
            bias for that both weight. 

            how to use: 

            import numpy as np 

            from LittleLearn import GradientReflector 

            x = np.random.normal(0,1,(10,5,32))

            units = 16 \n
            features = 32

            w_sequence = np.random.normal(0,1,(features,units))\n
            w_hidden = np.random.normal(0,1,(featuresmunits))

            bias = np.zeros((1,units))

            batch,seq,dim = x.shape

            hidden_state = np.zeros((batch,units))

            x = GradientReflector(x)

            x.simple_rnn_backend(hidden_state,weight=(w_sequence,w_hidden),bias_=bias) 

            Written by : Candra Alpin Gunawan 
        """

        x = self.tensor 
        batch,seq,dim = x.shape
        hiddensize = hidden_state.shape[-1]
        if len(weight) != 2 :
            raise RuntimeError("weight must 2 for sequence and hidden state")
        if bias_ is None :
            raise RuntimeError("bias must be Gradient reflector array or just array")
        w_sequance,w_hidden_s = weight
        bias_ = bias_ if isinstance(bias_,GradientReflector) else GradientReflector(bias_,_op="bias")

        w_sequance = w_sequance if isinstance(w_sequance,GradientReflector) else GradientReflector(w_sequance,_op="RNNWeight")
        w_hidden_s = w_hidden_s if isinstance(w_hidden_s,GradientReflector) else GradientReflector(w_hidden_s,_op="RNN_Hidden_s")
        hidden_state = hidden_state if isinstance(hidden_state,GradientReflector) else GradientReflector(hidden_state)
        
        def step(x,hs,weight_s,weight_h,b) :
            sequence_logits = np.matmul(x,weight_s) 
            hidden_logits = np.matmul(hs,weight_h)
            logits = (sequence_logits + hidden_logits) + b 
            return np.tanh(logits)
        
        hidd_his = list()
        x_hist = list()
        h_prev = list()

        def execution (x) :
            h = hidden_state.tensor
            h_prev.append(h)
            seq_out = list() 
            for i in range(x.shape[1]) :
                x_iter = x[:,i,:]
                x_hist.append(x_iter)
                h = step(x_iter,h,w_sequance.tensor,w_hidden_s.tensor,bias_.tensor)
                hidd_his.append(h)
                h_prev.append(h)
                seq_out.append(h)
            return np.stack(seq_out,axis=1,dtype=np.float32)
        out = execution(x)
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,hidden_state,w_sequance,w_hidden_s,bias_),_op='SimpleRNN')
        else :
             outputs = GradientReflector(out,_op='SimpleRNN')

        def backward () :
            d_xhist = np.zeros_like(w_sequance.tensor,dtype=np.float32)
            d_hiddst = np.zeros_like(w_hidden_s.tensor,dtype=np.float32)
            db = np.zeros_like(bias_.tensor,dtype=np.float32)
            d_next = np.zeros((batch,hiddensize),dtype=np.float32)
            d_x_next= list()
            for t in reversed(range(seq)) :
                dh = outputs.gradient[:,t,:] + d_next
                h = hidd_his[t]
                h_p = h_prev[t]
                xt = x_hist[t]

                dtanh  = (1 - h**2) * dh 
                d_xhist += np.dot(xt.T,dtanh)
                d_hiddst += np.dot(h_p.T,dtanh)
                db += np.sum(dtanh,axis=0,keepdims=True)
                dx = np.dot(dtanh,w_sequance.tensor.T)
                d_x_next.insert(0,dx)
                d_next =np.matmul(dtanh,w_hidden_s.tensor.T)

            w_sequance.gradient += d_xhist
            w_hidden_s.gradient += d_hiddst
            bias_.gradient += db 
            self.gradient += np.stack(d_x_next,axis=1)
        outputs._backwardpass = backward
        return outputs

    def embedding_back_end (self,weight) : 

        """
            embdding_back_end for Embedding layers Backend, its work for Embedding layers 
            forward and backward. 

            Embedding layers working to be matriks lookup for logits the reinpretitation token indeks. 

            how to use:

            import numpy as np\n 
            from LittleLearn import GradientReflector 

            data = np.array(([[10,2,3,1],[4,9,2,6]]))

            weight = np.random.rand(11,32) 

            data = GradientReflector(data)

            data.embedding_back_end(weight=weight)

            Written by: Candra Alpin Gunawan 
        """

        weight = weight if isinstance(weight,GradientReflector) else GradientReflector(weight,_op="weightembedding")
        x = self.tensor 
        out = weight.tensor[x]
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,weight),_op='embedding')
        else :
            outputs = GradientReflector(out,_op='embedding')
        def backward() :
            grad_weight = np.zeros_like(weight.tensor,dtype=np.float32)
            for i,token_id in enumerate(x.reshape(-1)) : 
                grad_weight[token_id] += outputs.gradient.reshape(-1,weight.tensor.shape[1])[i]

            weight.gradient += grad_weight
        outputs._backwardpass = backward 
        return outputs 
    
    def lstm_backend (self,hidden_state,cell_state,weight=(),bias=(),return_sequence=False,
                      return_state = False) :
        """
            lstm_backend is LSTM layers backend operation from Gradient Reflector. working for handle 
            LSTM layers operation.

            LSTM layers working by 4 weight and 4 bias that is forget gate,input gate,output gata,cell gate. 

            how to use:

            import numpy as np \n 
            from LittleLearn import GradientReflector \n 

            x = np.random.rand(10,15,32)

            units = 64 \n 
            features = 32 + units \n 

            w_forget = np.random.rand(features,units)\n
            w_input = np.random.rand(features,units)\n
            w_outputs = np.random.rand(features,units)\n
            w_cell = np.random.rand(features,units)\n 

            b_forget = np.zeros((1,units))\n
            b_input = np.zeros((1,units))\n 
            b_output = np.zeros((1,units))\n 
            b_cell = np.zeros((1,units))\n 

            batch,seq,dim = x.shape 

            hidden_state = np.zeros((batch,units))\n 
            cell_state = np.zeros((batch,units ))

            x = GradientReflector(x)

            x.lstm_backend(hidden_state,cell_state,weight=(w_forget,w_input,w_output,w_cell),
            bias=(b_forget,b_input,b_output,b_cell),return_sequence=False,return_state=False)

            Default return_sequence = False 

            Default return_state = False 

            if you want get by 3dim data set return_sequence to be True and if you want get hidden_state 
            that it hidden_state and cell_state set return_state = True 

            Reference:
            ----------
            - Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". Neural Computation.

            Author: Candra Alpin Gunawan

        """
        
        @jit(nopython=True,cache = True) 
        def sigmoid(x) :
            return 1 / (1 + np.exp(-x))
        
        @jit(nopython=True,cache=True)
        def tanh(x):
            return np.tanh(x)
        
        if len(weight) != 4 :
            raise RuntimeError("weight for lstm must be 4 its for (forget gate,cell gate,output gate, input gate)")
        if len(bias) !=4 :
            raise RuntimeError("bias for lstm must be 4 its for (forget gate,outputs gate,input gate,cell state gate)")
        
        x = self.tensor 
        batch,seq,dim = x.shape 
        hidden_state = hidden_state if isinstance(hidden_state,GradientReflector) else GradientReflector(hidden_state,_op='hiddenstate')
        cell_state = cell_state if isinstance(cell_state,GradientReflector) else GradientReflector(cell_state,_op='cellstate')

        w_forgot,w_input,w_output,w_cell = weight
        b_forgot,b_input,b_output,b_cell = bias 
        
        w_forgot = w_forgot if isinstance(w_forgot,GradientReflector) else GradientReflector(w_forgot,_op="w_forget")
        w_input = w_input if isinstance(w_input,GradientReflector) else GradientReflector(w_input,_op='w_input')
        w_output = w_output if isinstance(w_output,GradientReflector) else GradientReflector(w_output,_op='w_output')
        w_cell = w_cell if isinstance(w_cell,GradientReflector) else GradientReflector(w_cell,_op='w_cell')
        b_forgot = b_forgot if isinstance(b_forgot,GradientReflector) else GradientReflector(b_forgot,'b_forget')
        b_input = b_input if isinstance(b_input,GradientReflector) else GradientReflector(b_input,_op='b_input')
        b_output = b_output if isinstance(b_output,GradientReflector) else GradientReflector(b_output,_op='b_output')
        b_cell = b_cell if isinstance(b_cell,GradientReflector) else GradientReflector(b_cell,_op='b_cell')
        fg_logits = list()
        in_logits = list()
        out_logits = list()
        c_logits = list()
        c_cadidate_ = list()

        def execution (x_t,h_t,c_t) : 
            combine_input = np.concatenate([x_t,h_t],axis=-1,dtype=np.float32)
            forget_logits = np.matmul(combine_input,w_forgot.tensor) + b_forgot.tensor 
            input_logits = np.matmul(combine_input,w_input.tensor) + b_input.tensor 
            output_logits = np.matmul(combine_input,w_output.tensor) + b_output.tensor 
            cell_logits = np.matmul(combine_input,w_cell.tensor) + b_cell.tensor 

            c_logits.append(cell_logits)

            forget_gate = sigmoid(forget_logits)
            input_gate = sigmoid(input_logits)
            output_gate = sigmoid(output_logits)
            fg_logits.append(forget_gate)
            in_logits.append(input_gate)
            out_logits.append(output_gate)
            cell_candidate = tanh(cell_logits)
            c_cadidate_.append(cell_candidate)
            cs = forget_gate * c_t + input_gate * cell_candidate
            hs = output_gate * tanh(cs)

            return hs,cs
        
        Hidden_state = hidden_state.tensor.copy()
        Cell_state = cell_state.tensor.copy()
        hidden_s = list()
        cell_s = list()
        c_state = list()
        h_state= list()
        cell_s.append(Cell_state)
        hidden_s.append(Hidden_state)
        x_state = list()

        out = list()
        for t in range(seq) : 
            x_t = x[:,t,:]
            x_state.append(x_t)
            hidden_state.tensor,cell_state.tensor = execution(x_t,Hidden_state,Cell_state)
            hidden_s.append(hidden_state.tensor)
            h_state.append(hidden_state.tensor)
            cell_s.append(cell_state.tensor)
            c_state.append(cell_state.tensor)
            out.append(hidden_state.tensor)
        out = np.stack(out,axis=1)
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,hidden_state,cell_state,w_forgot,w_output,w_input,w_cell,b_forgot,b_output,b_input,b_cell),'LSTM')
        else :
            outputs = GradientReflector(out,_op='LSTM')

        def backward() :
            grad_out = outputs.gradient 
            dwf_g = np.zeros_like(w_forgot.tensor,dtype=np.float32)
            dwi_g = np.zeros_like(w_input.tensor,dtype=np.float32)
            dwo_g = np.zeros_like(w_output.tensor,dtype=np.float32)
            dwc_g = np.zeros_like(w_cell.tensor,dtype=np.float32)

            db_f = np.zeros_like(b_forgot.tensor,dtype=np.float32)
            db_in = np.zeros_like(b_input.tensor,dtype=np.float32)
            db_out = np.zeros_like(b_output.tensor,dtype=np.float32)
            db_cell = np.zeros_like(b_cell.tensor,dtype=np.float32)

            dh_next = np.zeros((batch,hidden_state.tensor.shape[-1]))
            dc_next = np.zeros((batch,cell_state.tensor.shape[-1]))

            dxs = [] 

            for t in reversed (range(seq))  : 
                if return_sequence is True :
                    dh = grad_out[:,t,:] + dh_next
                else :
                    dh = grad_out + dh_next if t == seq - 1 else dh_next.copy()
                dc = dc_next.copy()
                c_prev = cell_s[t]
                h_prev = hidden_s[t]
                x_t = x_state[t]
                
                do = dh * np.tanh(c_state[t])
                dc += dh * out_logits[t] * (1 - tanh(c_state[t])**2)
                df = dc * c_prev
                di = dc * c_cadidate_[t]
                d_cc = dc * in_logits[t]

                dfo = df * fg_logits[t] * (1 - fg_logits[t])
                dio = di * in_logits[t] * (1 - in_logits[t])
                doo = do * out_logits[t] * (1 - out_logits[t])
                dc_c = d_cc * (1- c_cadidate_[t]**2)

                dz = np.concatenate([dfo,dio,doo,dc_c],axis=1)
                combined_input = np.concatenate([x_t,h_prev],axis=-1)

                d_combined = np.matmul(dz,np.concatenate([w_forgot.tensor,w_input.tensor,w_output.tensor,w_cell.tensor],axis=1).T)
                dx = d_combined[:,:dim]
                dh_next = d_combined [:,dim:]
                dc_next = dc * fg_logits[t]
                dxs.insert(0,dx)

                dwf_g+= np.matmul(combined_input.T,dfo)
                dwi_g+= np.matmul(combined_input.T,dio)
                dwo_g += np.matmul(combined_input.T,doo)
                dwc_g += np.matmul(combined_input.T,dc_c)

                db_f += np.sum(dfo,axis=0,keepdims=True)
                db_in += np.sum(dio,axis=0,keepdims=True)
                db_out += np.sum(doo,axis=0,keepdims=True)
                db_cell += np.sum(dc_c,axis=0,keepdims=True)

            self.gradient += np.stack(dxs,axis=1)
            w_forgot.gradient += dwf_g
            w_input.gradient += dwi_g 
            w_output.gradient += dwo_g
            w_cell.gradient += dwc_g

            b_forgot.gradient += db_f
            b_input.gradient += db_in
            b_output.gradient += db_out
            b_cell.gradient += db_cell
            hidden_state.gradient += dh_next
            cell_state.gradient += dc_next

        outputs._backwardpass = backward 
        if return_sequence and not return_state:
            return outputs
        elif return_sequence and return_state:
            return outputs, hidden_state, cell_state
        elif not return_sequence and not return_state:
            outputs.tensor = outputs.tensor[:, -1, :]
            return outputs
        elif not return_sequence and return_state:
            outputs.tensor = outputs.tensor[:, -1, :]
            return outputs, hidden_state, cell_state

    def gru_backend(self,hidden_state,weight=(),bias=(),return_sequence=False,return_state = False) : 
        """
            Gru backend is a operation from GradientReflector it for handle forward and backward Gru layers.\n 
            Gru backend working by 3 weight that is update gate,reset gate,candidate gate. with hidden state that asumtion \n
            memories at layers, Gru can give a same effect like lstm but follow more effiency. 

            how to use:

            import numpy as np\n 
            from LittleLearn import GradientReflector\n 
            x = np.random.rand(10,15,32)\n 

            units = 16\n 
            features = 32 + units\n 
            w_update = np.random.rand(features,units)

            w_reset = np.random.rand(features,units)

            w_candidate = np.random.rand(features,units) 

            b_update = np.zeros((1,units))
            
            b_reset = np.zeros((1,units))

            b_candidate = np.zeros((1,units))

            batch,seq,dim= x.shape

            hidden_stata = np.zeros((batch,units))

            x = GradientReflector(x)

            x.gru_backend(hidden_state,weight=(w_update,w_reset,w_candidate),
            bias = (b_update,b_reset,b_candidate),return_sequence=False,return_state=True)

            return_sequence default is False 

            return_state default is False 

            if you want get 3 dim output set return_sequence to be True and if you want get hidden_state 
            set return_state to be True 

        """
        @jit(nopython=True,cache = True) 
        def sigmoid(x) :
            return 1 / (1 + np.exp(-x))
        
        @jit(nopython=True,cache=True)
        def tanh(x) :
            return np.tanh(x)
    
        hidden_state = hidden_state if isinstance(hidden_state,GradientReflector) else GradientReflector(hidden_state,_op='hiddenstate')
        x = self.tensor 
        batch,seq,dim = x.shape 
        w_up_gate,w_re_gate,w_candidate_gate = weight
        b_up_gate,b_re_gate,b_candidate_gate = bias 

        w_up_gate = w_up_gate if isinstance(w_up_gate,GradientReflector) else GradientReflector(w_up_gate,_op='W_update')
        w_re_gate = w_re_gate if isinstance(w_re_gate,GradientReflector) else GradientReflector(w_re_gate,_op='W_reset')
        w_candidate_gate = w_candidate_gate if isinstance(w_candidate_gate,GradientReflector) else GradientReflector(w_candidate_gate,_op='W_candidate')

        b_up_gate = b_up_gate if isinstance(b_up_gate,GradientReflector) else GradientReflector(b_up_gate,_op='B_update')
        b_re_gate = b_re_gate if isinstance(b_re_gate,GradientReflector) else GradientReflector(b_re_gate,_op='B_reset')
        b_candidate_gate = b_candidate_gate if isinstance(b_candidate_gate,GradientReflector) else GradientReflector(b_candidate_gate,_op='B_candidate')

        update_hist = list()
        reset_hist = list()
        candidate_hist = list()

        def execution(x_t,h_t) :
            combined_input = np.concatenate([x_t,h_t],axis=-1,dtype=np.float32)
            update_logits = np.matmul(combined_input,w_up_gate.tensor) + b_up_gate.tensor 
            reset_logits = np.matmul(combined_input,w_re_gate.tensor) + b_re_gate.tensor 

            update_g = sigmoid(update_logits)
            reset_g = sigmoid(reset_logits)
            update_hist.append(update_g)
            reset_hist.append(reset_g)

            candidate_combined = np.concatenate([x_t,(reset_g * h_t)],axis=-1,dtype=np.float32)
            
            candidate_logits = np.matmul(candidate_combined,w_candidate_gate.tensor ) + b_candidate_gate.tensor 
            candidate_g = tanh(candidate_logits)

            candidate_hist.append(candidate_g)

            hidden_s = (1 - update_g) * h_t + update_g * candidate_g

            return hidden_s
        
        hidden_prev = list()
        hidden_prev.append(hidden_state.tensor.copy())
        output = list()
        x_hist = list()
        for t in range(seq) : 
            x_t = x[:,t,:]
            x_hist.append(x_t.copy())
            hidden_state.tensor = execution(x_t,hidden_state.tensor)
            hidden_prev.append(hidden_state.tensor.copy())
            output.append(hidden_state.tensor)
        output = np.stack(output,axis=1)
        if self.active_grad is True:
            outputs = GradientReflector(output,(self,hidden_state,w_up_gate,w_re_gate,w_candidate_gate,b_up_gate,b_re_gate,b_candidate_gate),_op='GRU')
        else :
            outputs = GradientReflector(output,_op='GRU')
            
        def backward() :
            dw_up = np.zeros_like(w_up_gate.tensor,dtype=np.float32)
            dw_re = np.zeros_like(w_re_gate.tensor,dtype=np.float32)
            dw_candidate = np.zeros_like(w_candidate_gate.tensor,dtype=np.float32)

            db_up = np.zeros_like(b_up_gate.tensor,dtype=np.float32)
            db_re = np.zeros_like(b_re_gate.tensor,dtype=np.float32)
            db_candidate = np.zeros_like(b_candidate_gate.tensor,dtype=np.float32)

            dx = []
            dh_next = np.zeros((batch,hidden_state.tensor.shape[-1]),dtype=np.float32)
            grad_out = outputs.gradient 
            dim = hidden_state.shape[-1]
            dx_ndim = self.tensor.shape[-1]
            for t in reversed(range(seq)) : 
                if return_sequence is True : 
                    dh = grad_out[:,t,:] + dh_next 
                else :
                    dh = grad_out + dh_next if t == seq -1 else dh_next.copy()
                h_prev = hidden_prev[t]
                zt = update_hist[t]
                ct = candidate_hist[t]
                rt= reset_hist[t]
                x_t = x_hist[t]

                d_zt = dh * (ct - h_prev)
                dcand_t = dh * zt 
                dh_prev = dh * (1 - zt)

                dcand_raw = dcand_t * (1 - ct **2)
                combined_xh = np.concatenate([x_t,h_prev],axis=-1,dtype=np.float32)
                candidate_combined = np.concatenate([x_t,(rt * h_prev)],axis=-1,dtype=np.float32)

                dw_candidate += np.matmul(candidate_combined.T,dcand_raw) 
                db_candidate += np.sum(dcand_raw,axis=0)

                drh_prev = np.matmul(dcand_raw,w_candidate_gate.tensor.T)
                drh_for_re = drh_prev[:,-dim:]
                dr_t = drh_for_re * h_prev
                dh_prev += drh_prev[:,-dim:] * rt

                dr_raw = dr_t * rt * (1 - rt)
                dz_raw = d_zt * zt * (1 - zt)

                dw_up += np.matmul(combined_xh.T,dz_raw)
                db_up += np.sum(dz_raw,axis=0)
                dw_re += np.matmul(combined_xh.T,dr_raw)
                db_re += np.sum(dr_raw,axis=0)

                d_comb_up = np.matmul(dz_raw,w_up_gate.tensor.T)
                d_comb_re = np.matmul(dr_raw,w_re_gate.tensor.T)

                d_comb_cand = np.matmul(dcand_raw,w_candidate_gate.tensor.T)
                
                dh_prev += d_comb_up[:,-dim:] + d_comb_re[:,-dim:] + d_comb_cand[:,-dim:]

                dx_t = d_comb_up [:,:dx_ndim] + d_comb_re[:,:dx_ndim] + d_comb_cand[:,:dx_ndim]
                dx.insert(0,dx_t)

                dh_next = dh_prev.copy()
        
            hidden_state.gradient += dh_next
            self.gradient += np.stack(dx,axis=1)
            w_up_gate.gradient += dw_up 
            w_re_gate.gradient += dw_re
            w_candidate_gate.gradient += dw_candidate
            b_up_gate.gradient += db_up
            b_re_gate.gradient += db_re
            b_candidate_gate.gradient += db_candidate

        outputs._backwardpass = backward
        if return_sequence is True and return_state is False :
            return outputs
        
        if return_sequence is False and return_state is False : 
            outputs.tensor = outputs.tensor[:,-1,:]
            return outputs
        
        if return_sequence is True and return_state is True :
            return outputs,hidden_state
        
        if return_sequence is False and return_state is True  : 
            outputs.tensor = outputs.tensor[:,-1,:]
            return outputs,hidden_state
        
        return outputs
    
    def dense_backend (self,weight,bias,reduce_grad : Literal['mean','sum'] = 'sum') : 
        """
            Dense layers back end: 
            ------------------------

            parameter: 
            --------- 
                weight: np.array() default None 
                    its weight layers parameters 
                
                bias: np.array() default None 
                    its bias layers parameters 
                
                reduce_grad: optional [mean,sum] Default sum 
                    its reduce gradient methode when input > 2 dimention 

            how to use: 

            weight = np.random.rand(10,32)
            bias = np.random.rand(1,32)
            inputs = np.random.rand(100,10)

            inputs = GradientReflector(inputs)

            outputs = inputs.dense_backend(weight,bias,reduce_grad='sum')

            Author: Candra Alpin Gunawan 
        """


        if weight is None or bias is None :
            raise RuntimeError(f"weight/bias is {None} weight / bias must array ")

        x = self.tensor 
        weight = weight if isinstance(weight,GradientReflector) else GradientReflector(weight,_op='Dense_Weight')
        bias = bias if isinstance(bias,GradientReflector) else GradientReflector(bias,_op='Dense_Bias')

        out = np.matmul(x,weight.tensor) + bias.tensor 
        if self.active_grad is True:
            outputs = GradientReflector(out,(self,weight,bias),_op='Dense')
        else :
            outputs = GradientReflector(out,_op='Dense')

        def _backward() : 
            out_grad = outputs.gradient 
            inputs_hist = x.copy()
            if out_grad.ndim > 2 : 
                inputs_hist = x.swapaxes(-1,-2)
                axes = tuple(range(out_grad.ndim-2))
                axes_b = tuple(range(out_grad.ndim-1))
                if reduce_grad == 'sum' :
                    grad = np.matmul(inputs_hist,out_grad)
                    grad = np.sum(grad,axis=axes)
                    weight.gradient += grad 
                    bias.gradient += np.sum(out_grad,axis=axes_b)
                    self.gradient += np.matmul(out_grad,weight.tensor.T)
                elif reduce_grad == 'mean' :
                    grad = np.matmul(inputs_hist,out_grad)
                    grad = np.mean(grad,axis=axes)
                    weight.gradient += grad 
                    bias.gradient += np.mean(out_grad,axis=axes_b)
                    self.gradient += np.matmul(out_grad,weight.tensor.T)

                else :
                    raise RuntimeError("just available for sum and mean reduce method")
            else :
                grad = np.matmul(inputs_hist.T,out_grad)
                weight.gradient += grad 
                bias.gradient += np.sum(out_grad,axis=0)
                self.gradient += np.matmul(out_grad,weight.tensor.T)
        
        outputs._backwardpass = _backward
        return outputs 


    def expand_dims (self,axis) :
        x = self.tensor 
        out = np.expand_dims(x,axis=axis)
        outputs = GradientReflector(out,(self,),'add_dims')
        
        def _backward() :
            grad = outputs.gradient
            self.gradient +=np.reshape(grad,x.shape) 
        
        outputs._backwardpass = _backward
        return outputs
    
    def drop_out_backend(self,rate : float = 0.1) : 

        """
            drop out backend:
            -------------
            backend for drop out layers in DeepLearning.layers.Dropout, but you can 
            use it for general tensor too. 

            parameters:
            -------------
                rate: float
                    drop rate to look how many drop values for neuron 
            
            how to use:
            --------------
                x.drop_out_backend()
            
            return:
            ---------
            outputs : GradientReflector Tensor 

            Author: Candra Alpin Gunawan 
        """
        x = self.tensor
        p = 1 - rate
        m = np.random.binomial(1,p, size=(x.shape))
        drop_out = m * x / p
        outputs = GradientReflector(drop_out,(self,),_op='dropout')

        def backward() :
            out_grad = outputs.gradient 
            grad = m/p * out_grad
            self.gradient += grad 
        
        outputs._backwardpass = backward
        return outputs


    @property 
    def shape (self) :
        return self.tensor.shape 
    
    def ___Auto_gradient_exploaded_Detector(self, grad):
        if np.isnan(grad).any() or np.isinf(grad).any(): 
            raise RuntimeError("Training Stopped: Detected NaN or Inf in gradients. This is a critical explosion.")

        norm_values = np.linalg.norm(grad)
        self.__norm_signal = norm_values  

        if norm_values > 100:
            if self.__autoclip_log is True : 
                print(f"⚠️  Warning: Gradient exploded is Critics ! Norm = {norm_values:.4f} → Clip level 6")
            self.__grad_expload_signal = 0.25

        elif 90 <= norm_values <= 100:
            if self.__autoclip_log is True :
                print(f"⚠️  Warning: Gradient exploded! Norm = {norm_values:.4f} → Clip level 5")
            self.__grad_expload_signal = 0.5

        elif 75 <= norm_values < 90:
            if self.__autoclip_log is True:
                print(f"⚠️  Warning: Gradient high! Norm = {norm_values:.4f} → Clip level 4")
            self.__grad_expload_signal = 0.75

        elif 20 <= norm_values < 75:
            if self.__autoclip_log is True :
                print(f"⚠️  Info: Moderate gradient. Norm = {norm_values:.4f} → Clip level 3")
            self.__grad_expload_signal = 1.0

        elif 10 <= norm_values < 20:
            if self.__auto_clip is True :
                print(f"🔹 Info: Stable gradient. Norm = {norm_values:.4f} → Clip level 2")
            self.__grad_expload_signal = 1.5

        elif 5 <= norm_values < 10:
            if self.__autoclip_log is True :
                print(f"🔹 Info: Very stable gradient. Norm = {norm_values:.4f} → Clip level 1")
            self.__grad_expload_signal = 2.0
         
        if 1e-8 < norm_values < 1e-5:
            boost_factor = min(5.0 / norm_values, 1e4)
            if self.__autoclip_log is True : 
                print(f"🧊 Gradient vanishing detected! Norm = {norm_values:.8f} → Boost ×{boost_factor:.2f}")
            self.__grad_expload_signal = boost_factor
            self.__norm_signal = norm_values

    def AutoClipGradient (self,show_log  : bool = False) :
        """
        AutoClipGradient:
        -----------------
        This method dynamically regulates gradient flow during backpropagation.

        Despite the name "Clipping", this mechanism handles *both* exploding and vanishing gradients.

        - Exploding gradients are suppressed using norm-based scaling.
        - Vanishing gradients (with norm < 1e-5) are safely boosted.
        - Built-in safety checks handle NaN and Inf values.

        Inspired by real-world failures in deep networks, AutoClipGradient ensures robust learning by maintaining healthy gradient magnitudes automatically.

        AutoClipGradient work by 6 protocol level clipping: 

        - level 6 is when the threshold norm Gradient > 100. level 6 clip with 0.25 scale values\n 
        - level 5 is when the threshold norm Gradient beetween 90 and 100. level 5 clip with 0.75 scale values\n
        - level 4 is when the threshold norm Gradient 75 and <90 . level 4 clip with 1.0 scale values\n 
        - level 3 is when the threshold norm Gradient 20 and < 70 . level 3 clip with 1.25 scale values\n 
        - level 2 is when the threshold norm Gradient <20 and <10 . level 2 clip with scale 1.5 scale values\n 
        - level 1 is when the threshold norm Gradient < 10 and == 5 . level 1 clip with scale 2.0 scales values\n 

        how to use: 

        import numpy as np\n
        from LittleLearn import GradientReflector 

        a = GradientReflector(np.random.rand(10,32))\n
        b = GradientReflector(np.random.uniform(0.01,0.01,(32,16)))\n
        c = a.matmul(b)\n 
        d = c.relu()\n 
        d.AutoClipGradient()\n 
        d.backwardpass()
        
        Written by : Candra Alpin Gunawan 
        """
        self.__auto_clip = True 
        self.__autoclip_log = show_log
    def ___Auto_GradientClipping (self,grad) :
        signal_exploaded = self.__grad_expload_signal
        signal_norm_values = self.__norm_signal
        if signal_exploaded is not None and signal_norm_values is not None :
            if signal_norm_values > 1e-5:
                scale = signal_exploaded / signal_norm_values 
                return grad * scale 
            if signal_norm_values <= 1e-5 :
                return grad * signal_exploaded
        else :
            return grad 



    def backwardpass (self) :
        """
            call it for do backpropogation or backward pass to any operation that from GradientReflector operation.\n 

            how to use : 

            import numpy as np\n
            from LittleLearn import GradientReflector 

            a = GradientReflector(np.random.normal(0,0.1,(10,32)))\n
            b = GradientReflector(no.random.rand(32,64))\n
            c = a.matmul(b)\n 
            d = c.leaky_relu()\n 
            d.backwardpass()

            Written by : Candra Alpin Gunawan 
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child_ref in v._Node:
                    child = child_ref()
                    if child is not None:
                        build_topo(child)
                topo.append(v)

        build_topo(self)

        self.gradient = np.ones_like(self.tensor, dtype=self.dtype)

        _is_not_last_node = False
        for node in reversed(topo):
            node._backwardpass()
            if self.__auto_clip and _is_not_last_node:
                self.___Auto_gradient_exploaded_Detector(node.get_gradient())
                node.gradient = self.___Auto_GradientClipping(node.get_gradient())
            _is_not_last_node = True

        topo.clear()
        visited.clear()

    
    def kill_grad (self) :
        """
            call it for reset gradient when you need it.\n 
            i recomended call kill grad when you training layers by use raws layers.\n 
            or when you make layers that use Gradient Reflector for back end layers.\n

            how to use: 

            import numpy as np\n
            from LittleLearn import GradientReflector 

            a = GradientReflector(np.random.rand(10,32))\n
            b = GradientReflector(np.random.uniform(0.01,0.01,(32,16)))\n
            c = a.matmul(b)\n 
            d = c.relu()\n 
            d.backwardpass()\n 
            d.kill_grad()

            Written by : Candra Alpin Gunawan 
            
        """

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child_ref in v._Node:
                    child = child_ref()
                    if child is not None:
                        build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in topo:
            node.gradient = np.zeros_like(node.tensor, dtype=node.dtype)
        
        for node in topo :
            node._Node.clear()
            node._backwardpass = lambda: None

        topo.clear()
        visited.clear()
               
    def plot_trace_operation (self) :
        """
            call it for plot any opertion that conected with this operation . 

            how to use  : 

            a = GradientReflector(10)\n
            b = GradieentReflector(16)\n
            c = a + b \n
            c.backwardpass()\n
            c.plot_trace_operation()\n

            Written by : Candra Alpin Gunawan 
        """

        visited = set()
        G = nx.DiGraph()

        def build(Node) :
            if Node not in visited :
                visited.add(Node)
                if Node is not None : 
                    G.add_node(id(Node),label=Node._op)
                    for parent in Node._Node :
                        p = parent() 
                        G.add_edge(id(p),id(Node))
                        build(p)

        build(self)
        labels = nx.get_node_attributes(G,'label')
        pos = nx.spring_layout(G)
        nx.draw(G,pos,with_labels=True,labels=labels,node_color = 'lightblue',arrows=True)
        plt.title("Gradient Reflector tracked graph operation")
        plt.show()
  
