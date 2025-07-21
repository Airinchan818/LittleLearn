import numpy as np 
from numba import jit 
class Adam :
    """
        Adam (Adaptive Moment Estimation)

        Proposed in the paper: "Adam: A Method for Stochastic Optimization" 
        by Diederik P. Kingma and Jimmy Ba (2014).
        [https://arxiv.org/abs/1412.6980]

        Adam maintains two running averages for each parameter:
        - The first moment (mean of gradients): m_t
        - The second moment (uncentered variance of gradients): v_t

        These estimates are biased towards zero at initial steps, 
        so bias correction is applied:
            m_hat = m_t / (1 - beta1^t)
            v_hat = v_t / (1 - beta2^t)

        Parameter update rule:
            theta = theta - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

        Where:
        - beta1 is the exponential decay rate for the 1st moment estimates (default: 0.9)
        - beta2 is the exponential decay rate for the 2nd moment estimates (default: 0.999)
        - epsilon is a small constant to prevent division by zero (default: 1e-8)

        Adam is efficient in terms of computation, has little memory requirement,
        is invariant to diagonal rescaling of the gradients, and well suited for
        problems that are large in terms of data and/or parameters.

        how to use : 

            when train with Sequential layers wrapper : 
                from LittleLearn import DeepLearning as dl 

                model = dl.Model.Sequential([
                    dl.layers.Dense(32,activation = 'relu'),
                    dl.layers.Dense(1,activation = 'linear')
                ])

                model.build_model(optimizer = dl.optimizers.Adam(learning_rate = 1e-3),
                loss = dl.loss.MeanSquaredError())
            
            when train with raw layers : 

                from LittleLearn import DeepLearning as dl 
                
                x,y = training_datasets 

                layers1 = dl.layers.Dense(32,activation='relu')
                
                layers2 = dl.layers.Dense(1,activation='sigmoid')

                optimizer_fn = dl.optimizers.Adam()

                loss = dl.loss.BinaryCrossentropy()

                for epoch in range (10) : \n
                    x1 = layers1(x)\n
                    out = layers2(x2)\n
                    loss_ = loss(y,out)\n
                    loss_.AutoClipGradient()\n
                    loss_.backwardpass()
            
                    weight = [layers1.weight,layers1.bias,layers2.weight,layers2.bias]

                    optimizer_fn.apply_weight(weight)

                    optimizer.forward_in_weight()

                    loss_.kill_grad  
        Optimizer Implementation: Adam (Adaptive Moment Estimation)

        Written by: Candra Alpin Gunawan

        Inspired by: "Adam: A Method for Stochastic Optimization" 
                    by D. P. Kingma and J. Ba (2014)
                    [https://arxiv.org/abs/1412.6980]



    """


    def __init__ (self,learning_rate=0.001,Beta1=0.9,Beta2=0.999,epsilon=1e-5) :
        self.Beta1 =Beta1
        self.Beta2 = Beta2
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.Rmsprop_w = None 
        self.Rmsprop_b = None 
        self.learning_rate = learning_rate
        self.epsilon = epsilon 
        self.name = 'adam'
        self.model_weight = None
        self.iteration = 0
        self.Momentum = None
        self.Rmsprop = None 
        self.parameter = None 

    def build_component (self,features,bias) :
        self.Momentum_w = np.zeros_like(features)
        self.Momentum_b = np.zeros_like(bias)
        self.Rmsprop_w = np.zeros_like(features)
        self.Rmsprop_b = np.zeros_like(bias)
        count = 1 
        for i in self.Momentum.shape :
            count*=i 
        param = count + len(bias)
        self.parameter = param * 2 

    
    def build_from_weight (self) :
        try: 
            if self.model_weight is None :
                raise RuntimeError("the model_weight can't be None")
            if not isinstance(self.model_weight,list) :
                raise RuntimeError("the model weight have to in list")
            self.Momentum = list()
            self.Rmsprop = list()
            for i in range(len(self.model_weight)) :
                self.Momentum.append(np.zeros_like(self.model_weight[i].tensor))
                self.Rmsprop.append(np.zeros_like(self.model_weight[i].tensor))
            
            param = list() 
            for i in range(len(self.Momentum)) : 
                count = 1
                for n in self.Momentum[i].shape :  
                    count *= n 
                param.append(count)
            param = np.array(param)
            self.parameter = (np.sum(param)) * 2
 
        except Exception as e :
            e.add_note("Adam(model_weight=model.weight)")

    def __call__ (self,weight,bias,gradient_w,gradient_b) :
        if self.model_weight is None : 
            if self.Momentum_w is None or self.Rmsprop_w is None :
                self.build_component(weight,bias)
            iteration = self.iteration
            self.Momentum_w = self.Beta1 * self.Momentum_w + (1 - self.Beta1) * gradient_w
            self.Momentum_b = self.Beta1 * self.Momentum_b + (1 - self.Beta1) * gradient_b
            self.Rmsprop_w = self.Beta2 * self.Rmsprop_w + (1 - self.Beta2) * np.power(gradient_w,2)
            self.Rmsprop_b = self.Beta2 * self.Rmsprop_b + (1 - self.Beta2) * np.power(gradient_b,2)
            Momentum_w = self.Momentum_w / (1 - np.power(self.Beta1,(iteration + 1)))
            Momentum_b = self.Momentum_b / (1 - np.power(self.Beta1,(iteration + 1)))
            Rmsprop_w = self.Rmsprop_w / (1 - np.power(self.Beta2,(iteration + 1)))
            Rmsprop_b = self.Rmsprop_b / (1 - np.power(self.Beta2,(iteration + 1)))
            weight -= self.learning_rate / (np.sqrt(Rmsprop_w + self.epsilon)) * Momentum_w
            bias -= self.learning_rate / (np.sqrt(Rmsprop_b + self.epsilon)) * Momentum_b
            self.iteration +=1 
            return {
                'weight':weight,
                'bias':bias
            }
        else :
            raise RuntimeError("this method just run if model_weight is None, Run this optimizer with forward_in weight()")
    
    def ___update_weight(self,weight,grad,lr,epsilon,momentum,rms,beta1,beta2,iteration):

        @jit(nopython=True,cache=True)
        def count_grad (weight,grad,lr,epsilon,momentum,rms,beta1,beta2,iteration) :
            moment_bn = beta1 * momentum + (1 - beta1) * grad 
            moment_correct = moment_bn / (1 - np.power(beta1,(iteration + 1))) 
            rms_bn = beta2 * rms + (1 - beta2) * np.power(grad,2)
            rms_correct = rms_bn / (1 - np.power(beta2,(iteration + 1)))
            weight -= lr / np.sqrt(rms_correct + epsilon) * moment_correct
            return weight,moment_bn,rms_bn
        
        return count_grad(weight=weight,grad=grad,lr=lr,epsilon=epsilon,momentum=momentum,
                          rms=rms,beta1=beta1,beta2=beta2,iteration=iteration)
    
    def apply_weight (self,weight : list) :
        if not isinstance(weight,list) :
            raise RuntimeError("Weight must be in list type data")
        self.model_weight = weight 

    def forward_in_weight (self) :
        """
        call it when training with neural network
        """

        if self.model_weight is None  :
            raise RuntimeError("must call apply_weight first")

        if self.Momentum is None or self.Rmsprop is None :
            self.build_from_weight()
        beta1 = self.Beta1
        beta2 = self.Beta2 
        lr = self.learning_rate
        iteration = self.iteration
        epsilon = self.epsilon
  
        for i in range(len(self.model_weight)) :
            
            weight = self.model_weight[i]
            gradient = weight.gradient
            momentum = self.Momentum[i]
            rms = self.Rmsprop[i]
            self.model_weight[i].tensor,self.Momentum[i],self.Rmsprop[i] = self.___update_weight(
                weight= weight.tensor,
                grad=gradient,
                momentum=momentum,
                rms=rms,
                beta1 = beta1,beta2=beta2,
                lr = lr,epsilon=epsilon,
                iteration= iteration
            ) 
        self.iteration +=1 
    
class Rmsprop :

    """
        RMSprop (Root Mean Square Propagation)

        Proposed by Geoffrey Hinton in his Coursera lecture on Neural Networks (2012).
        Although it was not published in a peer-reviewed paper, it became widely adopted
        due to its effectiveness in non-stationary and online settings.

        RMSprop is designed to adapt the learning rate for each parameter individually.
        It keeps a moving average of the squared gradients to normalize the update step.

        The update rule for each parameter is:
            - Accumulate the squared gradients with exponential decay:
                E[g^2]_t = beta * E[g^2]_{t-1} + (1 - beta) * g_t^2

            - Then update the parameter:
                theta = theta - learning_rate * g_t / (sqrt(E[g^2]_t) + epsilon)

        Where:
        - g_t is the gradient at time step t
        - beta is the decay rate for the moving average (commonly 0.9)
        - epsilon is a small constant to avoid division by zero (e.g., 1e-8)

        RMSprop helps to reduce oscillations in the vertical direction, making it
        especially useful for non-convex problems such as training deep neural networks.
        It improves AdaGrad by preventing the learning rate from decaying too much.

           how to use : 

            when train with Sequential layers wrapper : 
                from LittleLearn import DeepLearning as dl 

                model = dl.Model.Sequential([
                    dl.layers.Dense(32,activation = 'relu'),
                    dl.layers.Dense(1,activation = 'linear')
                ])

                model.build_model(optimizer = dl.optimizers.Rmsprop(learning_rate = 1e-3),
                loss = dl.loss.MeanSquaredError())
            
            when train with raw layers : 

                from LittleLearn import DeepLearning as dl 
                
                x,y = training_datasets 

                layers1 = dl.layers.Dense(32,activation='relu')
                
                layers2 = dl.layers.Dense(1,activation='sigmoid')

                optimizer_fn = dl.optimizers.Rmsprop()

                loss = dl.loss.BinaryCrossentropy()

                for epoch in range (10) : \n
                    x1 = layers1(x)\n
                    out = layers2(x2)\n
                    loss_ = loss(y,out)\n
                    loss_.AutoClipGradient()\n
                    loss_.backwardpass()
            
                    weight = [layers1.weight,layers1.bias,layers2.weight,layers2.bias]

                    optimizer_fn.apply_weight(weight)

                    optimizer.forward_in_weight()

                    loss_.kill_grad  

        
        Written by: Candra Alpin Gunawan

        Originally proposed by: Geoffrey Hinton (2012) in his Coursera lecture
        (Not formally published in a peer-reviewed paper)
    """

    def __init__(self,learning_rate=0.001,Beta=0.999,epsilon=1e-5):
        self.Beta = Beta 
        self.rmsprop_w = None 
        self.rmsprop_b = None 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.name='rmspop'
        self.Rmsprop = None 
        self.model_weight = None 
        self.interation = 0
        self.parameter = None 
    
    def __build_component(self,features) :
        self.rmsprop_w = np.zeros((features,1))
        self.rmsprop_b = 0.0
    
    def __build_component_by_weight(self) :
        if self.model_weight is None : 
            raise RuntimeError("you have call apply_weight for run optimizer")
        if not isinstance(self.model_weight,list) :
            raise RuntimeError("The model weight must be list data ")
        self.Rmsprop = list()
        for w in self.model_weight :
            self.Rmsprop.append(np.zeros_like(w.tensor))
        
        param = list() 
        for i in range(len(self.Rmsprop)) :
            count = 1
            for n in self.Rmsprop[i].shape :
                count *= n 
            param.append(count)
        param = np.array(param)
        self.parameter = np.sum(param)
        
        

    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.rmsprop_w is None or self.rmsprop_b is None :
            self.__build_component(weight.shape[1])
        self.rmsprop_w = self.Beta * self.rmsprop_w + (1 - self.Beta) * np.power(gradient_w,2)
        self.rmsprop_b = self.Beta * self.rmsprop_b + (1 - self.Beta) * np.power(gradient_b,2)
        rmsprop_w = self.rmsprop_w / (1 - np.power(self.Beta,(iteration + 1)))
        rmsprop_b = self.rmsprop_b / (1 - np.power(self.Beta,(iteration + 1)))
        weight -= self.learning_rate / (np.sqrt(rmsprop_w + self.epsilon)) * gradient_w
        bias -= self.learning_rate / (np.sqrt(rmsprop_b + self.epsilon)) * gradient_b
        return {
            'weight' : weight,
            'bias' : bias
        }
    def __update_weight (self,weight,gradient,rms,lr,beta,iter,epsilon) :

        @jit(nopython=True,cache=True) 
        def count_grad(weight,gradient,rms,lr,beta,iter,epsilon) : 
            rms = beta * rms + (1 - beta) * np.power(gradient,2)
            rms_n = rms / ( 1 - np.power(beta,(iter + 1)))
            weight -= lr / np.sqrt(rms_n + epsilon) * gradient
            return weight,rms
        
        return count_grad(weight,gradient,rms,lr,beta,iter,epsilon)
    
    def forward_in_weight(self) : 
        """
            call it when training with neural network
        """
        if self.Rmsprop is None :
            self.__build_component_by_weight()

        iteration = self.interation
        epsilon = self.epsilon 
        lr = self.learning_rate
        beta = self.Beta
        for i in range(len(self.model_weight)) :
            weight = self.model_weight[i]
            grad = weight.gradient
            rms = self.Rmsprop[i]
            self.model_weight[i].tensor,self.Rmsprop[i] = self.__update_weight(
                weight=weight.tensor,gradient=grad,rms=rms,beta=beta,lr= lr ,
                iter = iteration,epsilon=epsilon
            )
        self.interation +=1 

    def apply_weight (self,weight : list) :
        if not isinstance(weight,list) :
            raise RuntimeError("weight must be list ")
        self.model_weight = weight 
    

    def change_component (self,Rmsprop_w,Rmspop_b) :
        self.rmsprop_w = Rmsprop_w
        self.rmsprop_b = Rmspop_b


class Momentum :
    """
        Momentum Optimizer

        Introduced in: "A Method for Accelerating the Convergence of Gradient Descent"
        by Polyak (1964), known as **Polyak's Heavy Ball Method**.

        Momentum accelerates gradient descent by taking into account
        the exponentially decaying moving average of past gradients,
        allowing faster convergence and reduced oscillation in directions
        of high curvature.

        The update rule involves maintaining a velocity vector `v_t`:
            v_t = beta * v_{t-1} - learning_rate * gradient
            theta = theta + v_t

        Where:
        - beta is the momentum coefficient (typically 0.9)
        - learning_rate is the step size
        - gradient is the current gradient of the loss with respect to parameters

        Interpretation:
        - The optimizer "builds up speed" in directions with consistent gradients
        - It helps to smooth out the noise and avoid getting stuck in local minima
        - Particularly effective in ravine-like loss surfaces (sharp curves in one dimension)

        Momentum is simple yet powerful and is often used as a component
        in more advanced optimizers like Nesterov and Adam.

           how to use : 

            when train with Sequential layers wrapper : 
                from LittleLearn import DeepLearning as dl 

                model = dl.Model.Sequential([
                    dl.layers.Dense(32,activation = 'relu'),
                    dl.layers.Dense(1,activation = 'linear')
                ])

                model.build_model(optimizer = dl.optimizers.Momentum(learning_rate = 1e-3),
                loss = dl.loss.MeanSquaredError())
            
            when train with raw layers : 

                from LittleLearn import DeepLearning as dl 
                
                x,y = training_datasets 

                layers1 = dl.layers.Dense(32,activation='relu')
                
                layers2 = dl.layers.Dense(1,activation='sigmoid')

                optimizer_fn = dl.optimizers.Momentum()

                loss = dl.loss.BinaryCrossentropy()

                for epoch in range (10) : \n
                    x1 = layers1(x)\n
                    out = layers2(x2)\n
                    loss_ = loss(y,out)\n
                    loss_.AutoClipGradient()\n
                    loss_.backwardpass()
            
                    weight = [layers1.weight,layers1.bias,layers2.weight,layers2.bias]

                    optimizer_fn.apply_weight(weight)

                    optimizer.forward_in_weight()

                    loss_.kill_grad  

        Written by: Candra Alpin Gunawan
        Inspired by: Boris Polyak, 1964
        "A Method for Accelerating the Convergence of Gradient Descent"
    """

    def __init__ (self,learning_rate=0.001,Beta=0.9) :
        self.Beta = Beta 
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.learning_rate = learning_rate
        self.name='momentum'
        self.Momentum = None 
        self.model_weight = None 
        self.iteration = 0 
        self.paramater = None 
    
    def __build_component (self,features) : 
        self.Momentum_w = np.zeros((features,1))
        self.Momentum_b = 0.0 
    
    def __build_in_weight(self) :
        if self.model_weight is None :
            RuntimeError("call apply_weight() first")
        
        self.Momentum = list()
        for w in self.model_weight :
            self.Momentum.append(np.zeros_like(w.tensor))
        
        param = list() 
        for i in range(len(self.Momentum)) :
            count = 1 
            for n in self.Momentum[i].shape :
                count *= n 
            param.append(count)
        param = np.array(param)
        self.paramater = np.sum(param)

    def forward_in_weight(self) :
        """call it when training with neural network"""
        if self.Momentum is None :
            self.__build_in_weight()
        beta = self.Beta
        lr = self.learning_rate
        iteration = self.iteration
        for i in range(len(self.model_weight)):
            weight = self.model_weight[i]
            moment = self.Momentum[i]
            grad = weight.gradient
            self.model_weight[i].tensor,self.Momentum[i] = self.__update_weight(
                weight=weight.tensor,grad=grad,momentum=moment,
                beta = beta ,lr = lr,iter=iteration
            )
        self.iteration +=1 
    

    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.Momentum_b is None :
            self.__build_component(weight.shape[1])
        self.Momentum_w = self.Beta * self.Momentum_w + (1 - self.Beta) * gradient_w
        self.Momentum_b = self.Beta * self.Momentum_b + (1 - self.Beta) * gradient_b
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta,(iteration + 1)))
        weight -= self.learning_rate * Momentum_w
        bias -= self.learning_rate * Momentum_b
        return {
            'weight' : weight,
            'bias' : bias
        }
    
    def __update_weight (self,weight,grad,momentum,beta,lr,iter) :

        @jit(nopython=True,cache=True) 
        def count_grad (weight,grad,momentum,beta,lr,iter) :
            momentum = beta * momentum + (1 - beta ) * grad 
            momentum_cr = momentum / (1 - np.power(beta,(iter + 1)))
            weight -= lr * momentum_cr
            return weight,momentum
        
        return count_grad(weight,grad,momentum,beta,lr,iter)

    def apply_weight (self,weight : list) :
        if not isinstance(weight,list) :
            raise RuntimeError("weight must be list type data ")
        self.model_weight = weight 

class GradientDescent :
    def __init__ (self,learning_rate=0.001) :
        self.learning_rate = learning_rate
        self.name ='gradient_descent'
    
    def __call__ (self,weight,bias,gradient_w,gradien_b) :
        weight -= self.learning_rate * gradient_w
        bias -= self.learning_rate * gradien_b
        return {
            'weight' : weight,
            'bias' : bias
        }

class AdamW :
    """
        AdamW (Adam with Decoupled Weight Decay)

        Proposed in the paper: "Decoupled Weight Decay Regularization"
        by Ilya Loshchilov and Frank Hutter (2017).
        [https://arxiv.org/abs/1711.05101]

        AdamW modifies the original Adam optimizer by **decoupling** weight decay
        from the gradient-based update. In standard Adam, weight decay is applied
        indirectly through L2 regularization, which can interfere with adaptive 
        moment estimates.

        In AdamW, weight decay is applied **directly** to the weights:
            theta = theta - learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * theta)

        Key differences from Adam:
        - Weight decay is applied independently, not as part of the loss gradient
        - Provides better generalization and performance, especially in large-scale training
        - Helps avoid the unintended regularization behavior in standard Adam

        Where:
        - m_hat and v_hat are bias-corrected estimates of the 1st and 2nd moments
        - weight_decay is a separate hyperparameter (e.g., 0.01 or 0.001)
        - epsilon prevents division by zero (typically 1e-8)

        AdamW is the default optimizer for many modern Transformer-based models (e.g., BERT, GPT)
        due to its robustness and better convergence behavior.


           how to use : 

            when train with Sequential layers wrapper : 
                from LittleLearn import DeepLearning as dl 

                model = dl.Model.Sequential([
                    dl.layers.Dense(32,activation = 'relu'),
                    dl.layers.Dense(1,activation = 'linear')
                ])

                model.build_model(optimizer = dl.optimizers.AdamW(learning_rate = 1e-3),
                loss = dl.loss.MeanSquaredError())
            
            when train with raw layers : 

                from LittleLearn import DeepLearning as dl 
                
                x,y = training_datasets 

                layers1 = dl.layers.Dense(32,activation='relu')
                
                layers2 = dl.layers.Dense(1,activation='sigmoid')

                optimizer_fn = dl.optimizers.AdamW()

                loss = dl.loss.BinaryCrossentropy()

                for epoch in range (10) : \n
                    x1 = layers1(x)\n
                    out = layers2(x2)\n
                    loss_ = loss(y,out)\n
                    loss_.AutoClipGradient()\n
                    loss_.backwardpass()
            
                    weight = [layers1.weight,layers1.bias,layers2.weight,layers2.bias]

                    optimizer_fn.apply_weight(weight)

                    optimizer.forward_in_weight()

                    loss_.kill_grad  

        Written by: Candra Alpin Gunawan

        Inspired by: "Decoupled Weight Decay Regularization"
             by Ilya Loshchilov and Frank Hutter (2017)
             [https://arxiv.org/abs/1711.05101]
    """

    def __init__ (self,learning_rate=1e-3,L2=4e-3,epsilon=1e-5,Beta1=0.9,Beta2=0.999) :
        self.learning_rate = learning_rate 
        self.L2 = L2
        self.epsilon = epsilon 
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.Rmsprop_w = None 
        self.Rmsprop_b = None
        self.Beta1 = 0.9 
        self.Beta2 = 0.999 
        self.name = 'adamw'
        self.model_weight = None 
        self.Momentum = None 
        self.Rmsprop = None 
        self.iteration = 0 
        self.parameter = None 

    def __build_component(self,features) :
        self.Momentum_w = np.zeros((features,1))
        self.Momentum_b = 0.0 
        self.Rmsprop_w = np.zeros((features,1))
        self.Rmsprop_b = 0.0
    
    def __build_in_weight(self) :
        self.Momentum = list()
        self.Rmsprop = list()
        
        for w in self.model_weight :
            self.Momentum.append(np.zeros_like(w.tensor))
            self.Rmsprop.append(np.zeros_like(w.tensor))
        
        param = list() 
        for i in range(len(self.Momentum)) :
            count = 1 
            for n in self.Momentum[i].shape :
                count *= n 
            param.append(count)
        
        param = np.array(param)
        self.parameter = (np.sum(param)) * 2 

    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.Rmsprop_w is None :
            self.__build_component(weight.shape[1])

        self.Momentum_w = self.Beta1 * self.Momentum_w + (1 - self.Beta1) * gradient_w
        self.Momentum_b = self.Beta1 * self.Momentum_b + (1 - self.Beta1) * gradient_b
        self.Rmsprop_w = self.Beta2 * self.Rmsprop_w + (1 - self.Beta2) * np.power(gradient_w,2)
        self.Rmsprop_b = self.Beta2 * self.Rmsprop_b + (1 - self.Beta2) * np.power(gradient_b,2)
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta1,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta1,(iteration + 1)))
        Rmsprop_w = self.Rmsprop_w / (1 - np.power(self.Beta2,(iteration + 1)))
        Rmsprop_b = self.Rmsprop_b / (1 - np.power(self.Beta2,(iteration + 1)))
        weight_decay = (1 - self.learning_rate * self.L2)
        bias_decay = (1 - self.learning_rate * self.L2)
        weight -= weight * weight_decay - self.learning_rate *  (Momentum_w / np.sqrt(Rmsprop_w + self.epsilon))
        bias -= bias * bias_decay - self.learning_rate / np.sqrt(Rmsprop_b + self.epsilon) * Momentum_b
        return {
            'weight' : weight,
            'bias' : bias
        }
    
    def forward_in_weight (self) :
        if self.Rmsprop is None or self.Momentum is None :
            self.__build_in_weight()
        
        beta1 = self.Beta1
        beta2 = self.Beta2
        epsilon = self.epsilon
        lr = self.learning_rate 
        L2 = self.L2
        iter = self.iteration

        for i in range(len(self.model_weight)) : 
            weight = self.model_weight[i]
            grad = weight.gradient 
            momentum = self.Momentum[i]
            rms= self.Rmsprop[i]
            self.model_weight[i].tensor,self.Momentum[i],self.Rmsprop[i] = self.__update_weight(
                weight=weight.tensor,grad=grad,moment=momentum,
                beta1=beta1,beta2=beta2,
                rms=rms,L2=L2,lr=lr,epsilon=epsilon,iter=iter
            )

        self.iteration +=1 



    def __update_weight(self,weight,grad,moment,rms,beta1,beta2,L2,lr,epsilon,iter) :

        @jit(nopython=True,cache=True)
        def count_grad (weight,grad,moment,rms,beta1,beta2,L2,lr,epsilon,iter) :
            momentum = beta1 * moment + (1 - beta1) * grad
            moment_corr = momentum / (1 - np.power(beta1,(iter + 1)))
            rmsprop = beta2 * rms + (1 - beta2) * np.power(grad,2)
            rms_correct = rmsprop / (1 - np.power(beta2,(iter + 1)))
            w_decay = (1 - lr * L2 )
            weight = weight * w_decay - lr * moment_corr / np.sqrt(rms_correct + epsilon) 
            return weight,momentum,rmsprop
        
        return count_grad(weight,grad,moment,rms,beta1,beta2,L2,lr,epsilon,iter)

    def apply_weight(self,weight : list) :
        if not isinstance(weight,list) :
            raise RuntimeError("weight must be list")
        self.model_weight = weight

    def change_component (self,Momentum_w,Momentum_b,rms_w,rms_b) :
        self.Momentum_w = Momentum_w
        self.Momentum_b = Momentum_b
        self.Rmsprop_w = rms_w
        self.Rmsprop_b = rms_b

class Adamax :

    """
        Adamax Optimizer

        Introduced as a variant of Adam in the original paper:
        "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014)
        [https://arxiv.org/abs/1412.6980]

        Adamax is a more stable version of Adam that uses the **infinity norm (max norm)**
        instead of the L2 norm for the second moment estimate. It is particularly useful
        when the gradients are sparse or when the L2-norm may become unstable.

        Core ideas:
        - Uses exponentially moving average of past gradients (m_t), like Adam
        - Replaces the second raw moment (v_t) with the exponentially weighted
        infinity norm (u_t), which is the maximum absolute gradient observed so far

        Update steps:
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            u_t = max(beta2 * u_{t-1}, |g_t|)
            theta = theta - (learning_rate / (u_t + epsilon)) * m_t

        Where:
        - g_t is the current gradient
        - m_t is the first moment (mean)
        - u_t is the exponentially weighted ∞-norm (max absolute value)
        - beta1, beta2 are decay rates (commonly 0.9 and 0.999)
        - epsilon is a small constant (e.g., 1e-8)

        Advantages:
        - More resilient to large gradient updates than standard Adam
        - Can be more stable in practice, especially in high-dimensional parameter spaces
        - Still retains Adam’s adaptive step size per parameter

        Adamax is not as widely used as Adam or AdamW, but may perform better
        in some cases involving very sparse or unbounded gradients.


           how to use : 

            when train with Sequential layers wrapper : 
                from LittleLearn import DeepLearning as dl 

                model = dl.Model.Sequential([
                    dl.layers.Dense(32,activation = 'relu'),
                    dl.layers.Dense(1,activation = 'linear')
                ])

                model.build_model(optimizer = dl.optimizers.Adamax(learning_rate = 1e-3),
                loss = dl.loss.MeanSquaredError())
            
            when train with raw layers : 

                from LittleLearn import DeepLearning as dl 
                
                x,y = training_datasets 

                layers1 = dl.layers.Dense(32,activation='relu')
                
                layers2 = dl.layers.Dense(1,activation='sigmoid')

                optimizer_fn = dl.optimizers.Adamax()

                loss = dl.loss.BinaryCrossentropy()

                for epoch in range (10) : \n
                    x1 = layers1(x)\n
                    out = layers2(x2)\n
                    loss_ = loss(y,out)\n
                    loss_.AutoClipGradient()\n
                    loss_.backwardpass()
            
                    weight = [layers1.weight,layers1.bias,layers2.weight,layers2.bias]

                    optimizer_fn.apply_weight(weight)

                    optimizer.forward_in_weight()

                    loss_.kill_grad  
       
            Optimizer implementation: Adamax

            Written by: Candra Alpin Gunawan

            Inspired by: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
            This implementation is original and built from scratch based on public research.
            No source code from external libraries was copied or reused

    """

    def __init__ (self,learning_rate = 1e-3,epsilon=1e-5,Beta1=0.9,Beta2=0.999) :
        self.learning_rate = learning_rate 
        self.epsilon = epsilon 
        self.Beta1 = Beta1
        self.Beta2 = Beta2 
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.abs_grad_w = None 
        self.abs_grad_b = None 
        self.name='adamax'
        self.model_weight = None 
        self.Momentum = None 
        self.MaxNorm = None 
        self.iteration = 0 
        self.parameter = None 

    
    def build_component (self,features) :
        self.Momentum_w = np.zeros((features,1))
        self.Momentum_b = 0.0 
        self.Rmsprop_w = np.zeros((features,1))
        self.Rmsprop_b = 0.0 
        self.parameter += len(self.Momentum_w[-1]) + len(self.Momentum_b[-1]) + len(self.Rmsprop_w[-1]) + len(self.Rmsprop_b[-1]) 

    def __build_in_weight(self) : 
        self.Momentum = list()
        self.MaxNorm = list()

        for w in self.model_weight :
            self.Momentum.append(np.zeros_like(w.tensor))
            self.MaxNorm.append(np.zeros_like(w.tensor))
        param = list()
        for i in range(len(self.Momentum)) : 
            count=1
            for n in self.Momentum[i].shape :
                count *=n 
            param.append(count)
        param = np.array(param)
        self.parameter = (np.sum(param)) * 2

    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.abs_grad_w is None :
            self.build_component(weight.shape[1])
        self.Momentum_w = self.Beta1 * self.Momentum_w + (1 - self.Beta1) * gradient_w
        self.Momentum_b = self.Beta1 * self.Momentum_b + (1 - self.Beta1) * gradient_b
        self.abs_grad_w = np.maximum((self.Beta2 * self.abs_grad_w),np.absolute(gradient_w))
        self.abs_grad_b = np.maximum((self.Beta2 * self.abs_grad_b),np.absolute(gradient_b))
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta1,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta1,(iteration + 1)))
        abs_grad_w = self.abs_grad_w
        abs_grad_b = self.abs_grad_b 
        weight -= self.learning_rate / (abs_grad_w + self.epsilon) * Momentum_w
        bias -= self.learning_rate / (abs_grad_b + self.epsilon) * Momentum_b
        return {
            'weight' : weight,
            'bias' : bias
        } 
    
    def forward_in_weight(self) :
        if self.Momentum is None or self.MaxNorm is None :
            self.__build_in_weight()

        beta1 = self.Beta1
        beta2 = self.Beta2
        lr = self.learning_rate
        iteration = self.iteration
        epsilon = self.epsilon

        for i in range(len(self.model_weight)) :
            weight = self.model_weight[i]
            grad = weight.gradient 
            moment = self.Momentum[i]
            max_norm = self.MaxNorm[i]
            self.model_weight[i].tensor,self.Momentum[i],self.MaxNorm[i] = self.__update_weight(
                weight = weight.tensor,grad = grad ,
                momentum=moment,Maxnorm=max_norm,
                beta1 = beta1,beta2 = beta2 , lr = lr , epsilon= epsilon,iter=iteration
            )
        
        self.iteration +=1 

    def apply_weight(self,weight : list) :
        if not isinstance(weight,list) :
            raise RuntimeError("weight must be list")
        
        self.model_weight = weight 
        
    def __update_weight (self,weight,grad,momentum,Maxnorm,beta1,beta2,lr,epsilon,iter) : 

        @jit(nopython=True,cache=True)
        def count_grad (weight,grad,momentum,Maxnorm,beta1,beta2,lr,epsilon,iter) :
            momentum = beta1 * momentum + (1 - beta1) * grad 
            moment_correct = momentum / (1 - np.power(beta1,(iter + 1)))
            max_norm_abs = np.maximum((beta2 * Maxnorm),np.abs(grad))

            weight -= lr / (max_norm_abs + epsilon) * moment_correct
            return weight,momentum,max_norm_abs
        
        return count_grad(weight,grad,momentum,Maxnorm,beta1,beta2,lr,epsilon,iter)
        

    def change_component (self,Momentum_w,Momentum_b,rms_w,rms_b) :
        self.Momentum_w = Momentum_w
        self.Momentum_b = Momentum_b
        self.Rmsprop_w = rms_w
        self.Rmsprop_b = rms_b


class Lion:
    """
        Lion (EvoLved Sign Momentum)

        Proposed in the paper: "Symbolic Discovery of Optimization Algorithms"
        by Chen et al., Google DeepMind (2023).
        [https://arxiv.org/abs/2302.06675]

        Lion is a **lightweight optimizer** that replaces expensive adaptive operations 
        (like sqrt and division in Adam) with simple **sign-based updates**, while still 
        achieving competitive or superior performance.

        It uses momentum like Adam, but updates weights using only the sign of the 
        momentum term, leading to faster and more memory-efficient optimization.

        Update rule:
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            theta = theta - learning_rate * sign(m_t)

        Optionally with weight decay:
            theta = (1 - weight_decay) * theta - learning_rate * sign(m_t)

        Where:
        - g_t is the current gradient
        - m_t is the exponential moving average of past gradients (first moment)
        - beta1 is the momentum decay (e.g., 0.9 or 0.95)
        - weight_decay is a multiplicative decay factor (decoupled from gradient)

        Key Features:
        - Avoids second moment computation (unlike Adam)
        - Uses only sign and momentum, making it efficient on modern hardware (e.g., TPU)
        - Comparable or better generalization performance than AdamW on many tasks
        - Especially useful in large-scale models such as Vision Transformers and LLMs

        Lion is often considered a modern alternative to AdamW,
        offering simpler math and improved speed without sacrificing convergence.


           how to use : 

            when train with Sequential layers wrapper : 
                from LittleLearn import DeepLearning as dl 

                model = dl.Model.Sequential([
                    dl.layers.Dense(32,activation = 'relu'),
                    dl.layers.Dense(1,activation = 'linear')
                ])

                model.build_model(optimizer = dl.optimizers.Lion(learning_rate = 1e-3),
                loss = dl.loss.MeanSquaredError())
            
            when train with raw layers : 

                from LittleLearn import DeepLearning as dl 
                
                x,y = training_datasets 

                layers1 = dl.layers.Dense(32,activation='relu')
                
                layers2 = dl.layers.Dense(1,activation='sigmoid')

                optimizer_fn = dl.optimizers.Lion()

                loss = dl.loss.BinaryCrossentropy()

                for epoch in range (10) : \n
                    x1 = layers1(x)\n
                    out = layers2(x2)\n
                    loss_ = loss(y,out)\n
                    loss_.AutoClipGradient()\n
                    loss_.backwardpass()
            
                    weight = [layers1.weight,layers1.bias,layers2.weight,layers2.bias]

                    optimizer_fn.apply_weight(weight)

                    optimizer.forward_in_weight()

                    loss_.kill_grad  

        Written by: [Candra Alpin Gunawan]

        Inspired by: "Symbolic Discovery of Optimization Algorithms"
             by Chen et al., DeepMind (2023)
             [https://arxiv.org/abs/2302.06675]
    """

    def __init__ (self, learning_rate = 1e-3, beta = 0.9) :

        self.learning_rate = learning_rate
        self.beta = beta 
        self.Momentum = None 
        self.model_weight = None 
        self.parameter = None 
    
    def __build_in_weight(self) :
        self.Momentum = list() 
        for w in self.model_weight : 
            self.Momentum.append(np.zeros_like(w.tensor))
        param = list()
        for i in range(len(self.Momentum)) :
            count = 1 
            for n in self.Momentum[i].shape :
                count*= n 
            param.append(count)
        param = np.array(param)
        self.parameter = np.sum(param) 
    
    def __update_weight(self,weight,grad,momentum,beta,lr) : 
        momentum = beta * momentum + (1 - beta) * grad 
        weight -= lr * np.sign(momentum)
        return weight,momentum
    
    def apply_weight (self,weight : list) :
        if not isinstance(weight,list) :
            raise RuntimeError("weight must be list")
        self.model_weight = weight 
    
    def forward_in_weight(self) :
        if self.Momentum is None :
            self.__build_in_weight()
        
        for i in range(len(self.model_weight)) :

            self.model_weight[i].tensor,self.Momentum[i] = self.__update_weight(
                weight=self.model_weight[i].tensor,
                grad=self.model_weight[i].gradient,
                momentum=self.Momentum[i],
                beta = self.beta,
                lr = self.learning_rate)