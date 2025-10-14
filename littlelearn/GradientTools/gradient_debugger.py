from littlelearn import GradientReflector 
import matplotlib.pyplot as plt 
from typing import Literal
import numpy as np 
from scipy.stats import gaussian_kde

def gradient_std_viewers (Node : list) :

    """
        gradient_std_viewers for look bar ploting any standar deviasi at 
        gradient. we wanna look, how many magnitude gradient in std values. 

        parameters:
            Node : list 
                list of gradient reflector tensor object

        how to use:
            gradient_std_viewers(model.get_weight())
        
        author: Candra Alpin Gunawan 
    """

    if not isinstance(Node[0],GradientReflector) :
        raise RuntimeError("Node in list must be Gradient Reflector Tensor object")
    Node_name = []
    Node_std = []
    for weight in Node :
        if isinstance(weight,GradientReflector) :
            Node_name.append(weight._op)
            Node_std.append(np.std(weight.get_gradient()))
    
    plt.title("Gradient Standard Deviasi")
    plt.xlabel("Name Node")
    plt.ylabel("Std Values")
    plt.bar(Node_name,Node_std)
    plt.show()

def gradient_outliners_scatter(Node : GradientReflector, mode: Literal['z_score','IQR','KDE'],
                               kde_threshold : int = 10):
    """
        gradient outliners scatter:
        --------------------------
        look for outliners gradient in Node paramers

        parameters:
        --------------
            Node : GradientReflector Tensor Object 
                Tensor Object in Grad engine Gradient Reflector 
            
            mode :optional('z_score','IQR','KDE)
                outliners detection counting method
        
        how to use:
        -----------
        radient_outliners_scatter(Node,mode='IQR')

        Author : Candra Alpin Gunawan

    """
    if not hasattr(Node, "get_gradient"):
        raise RuntimeError("Parameters Node must be GradientReflector-like object")
    grad = Node.get_gradient()

    if np.sum(grad) == 0:
        raise RuntimeError("gradient is zeros values, do backwardpass for get valid gradient")
    
    grad = np.array(grad)
    grad_flat = grad.flatten()

    plt.title(f"Outlier Detection (mode={mode})")
    plt.xlabel("Index")
    plt.ylabel("Gradient values")

    if mode == 'z_score':
        mean = np.mean(grad_flat)
        std = np.std(grad_flat)
        z_scores = (grad_flat - mean) / std
        outliers_mask = np.abs(z_scores) > 3
        outliers = grad_flat[outliers_mask]

    elif mode == 'IQR':
        Q1, Q3 = np.percentile(grad_flat, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask = (grad_flat < lower_bound) | (grad_flat > upper_bound)
        outliers = grad_flat[outliers_mask]

    elif mode == 'KDE':
        kde = gaussian_kde(grad_flat)
        density = kde.evaluate(grad_flat)
        threshold = np.percentile(density, kde_threshold)
        outliers_mask = density < threshold
        outliers = grad_flat[outliers_mask]
    else:
        raise ValueError("mode must be one of ['z_score', 'IQR', 'KDE']")

    plt.scatter(np.arange(len(grad_flat)), grad_flat, color='green', label='gradient')
    plt.scatter(np.arange(len(grad_flat))[outliers_mask], outliers, color='red', label='outliers')
    plt.legend()
    plt.grid(True)
    plt.show()

def gradient_Density (Node : GradientReflector) :
    """
        gradient_Density:
        -------------------
        counting by gaussian Density formula. this function created to look density 
        in gradient. 

        how to use:

        gradient_Density(Node)

        returns:

        outputs: density values of some gradient node

        author:

        Candra Alpin gunawan 
    """
    if not isinstance(Node,GradientReflector) :
        raise RuntimeError("this funtion just support for Gradient Reflector Tensor")
    if Node.gradient.sum() == 0:
        raise RuntimeError("gradient is zeros values, do backwardpass for get valid gradient")
    
    grad = Node.get_gradient()
    if grad.ndim>1 :
        grad = grad.flatten()
    
    kde = gaussian_kde(grad)
    return kde.evaluate(grad)

def gradient_effect_viewers (Node : GradientReflector,
                             op_effect : Literal['add','substract','multiple','divide'] = 'substract') :
    
    """
        gradient effect viewers:
        ------------------------
        use it for look how many effect gradient to tensor, gradient effect assumtion 
        for do some operation to tensor and gradient as factor operation. 

        parameters:
        --------------
        Node : GradientReflector
            Gradient Reflector tensor object 
        
        op_effect : optional ('add','substract','multiple','divide') default = 'substract'
            operation effect to showing in scatter 
        
        how to use:
        -------------
        gradient_effect_viewers (Node,op_effect='substract')

        Author
        ----------- 
        Candra Alpin Gunawan
    """
    if not isinstance(Node,GradientReflector) :
        raise RuntimeError("Node parameters must be Gradient Reflector Tensor Object")
    
    
    tensor = Node.get_tensor()
    grad = Node.get_gradient()
    subtract_effect = tensor - grad 
    add_effect = tensor + grad 
    multiple_effect = tensor * grad 
    divide_effect = tensor / grad 

    print(f"report :\nsubtract effect : {subtract_effect.mean()}\nadd effect : {add_effect}\nmultiple effect : {multiple_effect}\ndivide effect : {divide_effect}" )
    plt.title(f"gradient effect at operation :{op_effect} ")
    plt.xlabel("range of data")
    plt.ylabel("values of gradient")
    grad = grad.flatten()
    if op_effect == 'add' :
        add_effect = add_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]), grad, color = 'green',
                    label='gradient')
        plt.scatter(np.arange(add_effect.shape[0]),add_effect,
                    color='blue',label='gradient effect')
        
    elif op_effect == 'substract' :
        subtract_effect = subtract_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]),grad,color='green',label='gradient')
        plt.scatter(np.arange(subtract_effect.shape[0]),subtract_effect,
                    color='blue',label='gradient effect')
    
    elif op_effect == 'multiple' :
        multiple_effect = multiple_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]),grad,
                    label='gradient',color='green')
        plt.scatter(np.arange(multiple_effect.shape[0]),multiple_effect,
                    label='gradient effect',color='blue')
    
    elif op_effect == 'divide' :
        divide_effect = divide_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]),grad,
                    color='green',label='gradient')
        plt.scatter(np.arange(divide_effect.shape[0]),divide_effect,
                    label='gradient effect',color='blue')
    
    else :
        raise RuntimeError(f"op_effect : {op_effect} not aivable")
    plt.legend()
    plt.grid(True)
    plt.show()

def gradient_effect_factor_viewers(Node : GradientReflector, factor : float , 
                                   op_effect : Literal['add','subtract','multiple','divide'] = 'subtract'):
    """
        gradient effect viewers:
        ------------------------
        use it for look how many effect gradient to tensor, gradient effect assumtion that is
        for do some operation to tensor and (factor * gradient) as factor operation. 

        parameters:
        --------------
        Node : GradientReflector
            Gradient Reflector tensor object 
        
        factor : float (0.01/0.99)
            is factor scaling at gradient 
        
        op_effect : optional ('add','substract','multiple','divide') default = 'substract'
            operation effect to showing in scatter 
        
        how to use:
        -------------
        gradient_effect_viewers (Node,op_effect='substract')

        Author
        ----------- 
        Candra Alpin Gunawan
    """
    if not isinstance(Node,GradientReflector) :
        raise RuntimeError("Node parameters must be Gradient Reflector Tensor Object")
    if factor >=1.0 and factor < 0:
        raise RuntimeError(f"this values ({factor}) its abnormal for be factor gradient effect viewers")
    
    tensor = Node.get_tensor()
    grad = Node.get_gradient()

    add_effect = tensor + (factor * grad)
    subtract_effect = tensor - (factor * grad)
    divide_effect = tensor / (factor * grad)
    multiple_effect = tensor * (factor * grad)

    print(f"report :\nsubtract effect : {subtract_effect.mean()}\nadd effect : {add_effect}\nmultiple effect : {multiple_effect}\ndivide effect : {divide_effect}" )
    plt.title(f"gradient effect at operation :{op_effect} ")
    plt.xlabel("range of data")
    plt.ylabel("values of gradient")
    grad = grad.flatten()
    if op_effect == 'add' :
        add_effect = add_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]), grad, color = 'green',
                    label='gradient')
        plt.scatter(np.arange(add_effect.shape[0]),add_effect,
                    color='blue',label='gradient effect')
        
    elif op_effect == 'substract' :
        subtract_effect = subtract_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]),grad,color='green',label='gradient')
        plt.scatter(np.arange(subtract_effect.shape[0]),subtract_effect,
                    color='blue',label='gradient effect')
    
    elif op_effect == 'multiple' :
        multiple_effect = multiple_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]),grad,
                    label='gradient',color='green')
        plt.scatter(np.arange(multiple_effect.shape[0]),multiple_effect,
                    label='gradient effect',color='blue')
    
    elif op_effect == 'divide' :
        divide_effect = divide_effect.flatten()
        plt.scatter(np.arange(grad.shape[0]),grad,
                    color='green',label='gradient')
        plt.scatter(np.arange(divide_effect.shape[0]),divide_effect,
                    label='gradient effect',color='blue')
    
    else :
        raise RuntimeError(f"op_effect : {op_effect} not aivable")
    plt.legend()
    plt.grid(True)
    plt.show()


def gradient_mean_viewers (Node : list) :

    """
        gradient_std_viewers for look bar mean at 
        gradient. we wanna look, how many magnitude gradient in std values. 

        parameters:
            Node : list 
                list of gradient reflector tensor object

        how to use:
            gradient_mean_viewers(model.get_weight())
        
        author: Candra Alpin Gunawan 
    """

    if not isinstance(Node[0],GradientReflector) :
        raise RuntimeError("Node in list must be Gradient Reflector Tensor object")
    Node_name = []
    Node_std = []
    for weight in Node :
        if isinstance(weight,GradientReflector) :
            Node_name.append(weight._op)
            Node_std.append(np.mean(weight.get_gradient()))
    
    plt.title("Gradient Standard Deviasi")
    plt.xlabel("Name Node")
    plt.ylabel("Mean Values")
    plt.bar(Node_name,Node_std)
    plt.show()

def gradient_variant_viewers (Node : list) :

    """
        gradient_std_viewers for look bar varian at 
        gradient. we wanna look, how many magnitude gradient in std values. 

        parameters:
            Node : list 
                list of gradient reflector tensor object

        how to use:
            gradient_variant_viewers(model.get_weight())
        
        author: Candra Alpin Gunawan 
    """

    if not isinstance(Node[0],GradientReflector) :
        raise RuntimeError("Node in list must be Gradient Reflector Tensor object")
    Node_name = []
    Node_std = []
    for weight in Node :
        if isinstance(weight,GradientReflector) :
            Node_name.append(weight._op)
            Node_std.append(np.var(weight.get_gradient()))
    
    plt.title("Gradient Standard Deviasi")
    plt.xlabel("Name Node")
    plt.ylabel("variant Values")
    plt.bar(Node_name,Node_std)
    plt.show()