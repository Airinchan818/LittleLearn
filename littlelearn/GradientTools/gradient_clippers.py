from littlelearn import GradientReflector
import matplotlib.pyplot as plt 
import numpy as np 
from typing import Literal

class ClipByNorm:
    """
        ClipByNorm:
        -------------
        gradient Clipper By norm looking at Norm values per object, count norm values 
        and create scale values by maxnorm as threshold values factor and scaling to gradient 
        values. 

        parameters:
        ------------
        Model_weight : List( (Gradient Reflector Tensor))
            weight of model as Gradient Reflector Tensor object 
        
        maxnorm : float 
            threshold values as scaling factor to clipping gradient 
        
        reduce_hits : optional ('mean' or 'sum')
            reduce method for look gradient magnitude 
        
        example:
        -------------
            x_train,y_train = datasets and target 
            model = your_model() 
            loss_fn = your_loss_fn()
            optimizers = your_optimizer(model.get_weight()) 
            optimizers.apply_weight(model.get_weight())
            clipper = ClipByNorm(model.get_weight()) 
            for epoch in range(100) :
                outputs = model(x_train)
                loss = loss_fn(y_train,outputs)
                loss.backwardpass()
                clipper.execute()
                optimizers.forward_in_weight()
                loss.kill_grad()
                

        
        Author:
        --------
        Candra Alpin Gunawan 

    """
    def __init__ (self,Model_weight :list,max_norm : float ,reduce_hist : Literal['mean','sum']='mean') :
        self.max_norm = max_norm 
        self.magnitude_historys = []
        self.reduce_method = reduce_hist
        self.model_node = Model_weight
    
    def execute (self) :
        total_norm = 0 
        for weight in self.model_node :
            if isinstance(weight,GradientReflector) :
                grad = weight.get_gradient()
                norm = np.sqrt(np.sum(grad**2))
                total_norm += norm 
                if norm > self.max_norm :
                    scale = self.max_norm / norm 
                    grad *= scale 
                    weight.gradient = grad 
        
        if self.reduce_method == 'mean' :
            self.magnitude_historys.append(total_norm / len(self.model_node))
        elif self.reduce_method == 'sum' :
            self.magnitude_historys.append(total_norm)
        else :
            raise RuntimeError("reduce method just aivable for 'mean' or 'sum'")
    
    def plot_gradient_magnitude (self) :
        if len(self.magnitude_historys) == 0:
            raise RuntimeError("you must use clipper to look gradient magnitude")
        
        plt.title(f"Gradient Magnitude /n reduce method : {self.reduce_method}")
        plt.xlabel("Timesteps")
        plt.ylabel("Magnitudes values")
        plt.plot(self.magnitude_historys,label='gradient',color='green')
        plt.legend()
        plt.grid(True)
        plt.show()

class ClipByValues :

    """
        Clip By Values
        ---------------
        Gradient Clipper by values look to - clip values and clip values as 
        threshold values for clipping. 

        parameters:
        ----------------
        Model_weight : List(Gradient Reflector Tensor)
            weight of model as Gradient Reflector Tensor object 
        
        clip_values : float 
            clip_values a threshold for extreem negative values and positive values
        
        reduce_method : optional ('sum' or 'mean')
            reduce method for look gradient magnitude 

        example:
        -------------
            x_train,y_train = datasets and target 
            model = your_model() 
            loss_fn = your_loss_fn()
            optimizers = your_optimizer(model.get_weight()) 
            optimizers.apply_weight(model.get_weight())
            clipper = ClipByValues(model.get_weight()) 
            for epoch in range(100) :
                outputs = model(x_train)
                loss = loss_fn(y_train,outputs)
                loss.backwardpass()
                clipper.execute()
                optimizers.forward_in_weight()
                loss.kill_grad()
        Author:
        --------
        Candra Alpin Gunawan 

    """
    def __init__ (self,Model_weight : list,clip_values : float, 
                  reduce_method : Literal['sum','mean'] = 'mean') :
        self.model_node = Model_weight
        self.clip_values = clip_values
        self.reduce_method = reduce_method
        self.magnitude_hist = []
    
    def execute (self) :
        total_magnitude = 0 
        for weight in self.model_node :
            if isinstance(weight,GradientReflector) :
                grad = weight.get_gradient()
                if self.reduce_method == 'mean' :
                    total_magnitude += grad.mean()
                elif self.reduce_method == 'sum' :
                    total_magnitude += grad.sum()
                else :
                    raise RuntimeError("reduce method just aivable for 'mean' or 'sum'")
                grad = np.clip(grad,-self.clip_values,self.clip_values)
                weight.gradient = grad 
        
        if self.reduce_method == 'mean' :
            self.magnitude_hist.append(total_magnitude / len(self.model_node))
        elif self.reduce_method == 'sum' :
            self.magnitude_hist.append(total_magnitude)
    
    def plot_gradient_magnitude(self) :
        if len(self.model_node) == 0 :
            raise RuntimeError("you must use clipper to look gradient magnitude")

        plt.title(f"Gradient Magnitude \n reduce method : {self.reduce_method}")
        plt.xlabel("Timesteps")
        plt.ylabel("Magnitude values")
        plt.plot(self.magnitude_hist,label='gradient',color='green')
        plt.legend()
        plt.grid(True)
        plt.show()

class ClipByGlobalNorm :
    """
        ClipByGlobalNorm
        ------------------
        Gradient Clipper by Global Norm values look to - clip values by Global 
        Norm at Model weight. 

        parameters:
        -----------
        
        Model_weight : list(Gradient Reflector Tensor object)
            Model weight or node with Gradient Reflector Object 
        
        maxglobalnorm : float 
            threshold factor values for look at gradient norm 
        
        reduce_method : optional ('sum','mean')
            reduce method for look gradient magnitude 
        
        example:
        -------------
            x_train,y_train = datasets and target 
            model = your_model() 
            loss_fn = your_loss_fn()
            optimizers = your_optimizer(model.get_weight()) 
            optimizers.apply_weight(model.get_weight())
            clipper = ClipByGlobalNorm(model.get_weight()) 
            for epoch in range(100) :
                outputs = model(x_train)
                loss = loss_fn(y_train,outputs)
                loss.backwardpass()
                clipper.execute()
                optimizers.forward_in_weight()
                loss.kill_grad()
        Author:
        -----------
        Candra Alpin Gunawan 

    """
    def __init__ (self,Model_weight : list, maxglobalnorm : float,
                  reduce_method : Literal['sum','mean'] = 'mean') :
        self.model_node = Model_weight
        self.maxnorm = maxglobalnorm
        self.reduce_method = reduce_method
        self.magnitude_hist = []
    
    def execute(self) :
        norm_values = 0 
        for weight in self.model_node :
            if isinstance(weight,GradientReflector) :
                grad = weight.get_gradient()
                norm_values += np.sqrt(np.sum(grad**2))
        if norm_values > self.maxnorm :
            scale_factor = self.maxnorm / np.max(norm_values)
            for weight in self.model_node :
                if isinstance(weight,GradientReflector) :
                    grad = weight.get_gradient() * scale_factor
                    weight.gradient = grad 
        
        if self.reduce_method == 'mean' :
            self.magnitude_hist.append(norm_values / len(self.model_node))
        elif self.reduce_method == 'sum' :
            self.magnitude_hist.append(norm_values)
    
    def plot_gradient_magnitude (self) :

        if len(self.model_node) == 0 :
            raise RuntimeError("you must use clipper to look gradient magnitude")
        
        plt.title(f"Gradient magnitude \n reduce methode : {self.reduce_method}")
        plt.xlabel("Timesteps")
        plt.ylabel("Magnitude values")
        plt.plot(self.magnitude_hist,color='green',label='gradient')
        plt.legend()
        plt.grid(True)
        plt.show()
