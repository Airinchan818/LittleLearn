from typing import Literal
import littlelearn as ll 
class MSELoss:
    """
    MeanSquareError (MSE)
    ---------------------
    This class implements the Mean Squared Error (MSE) loss function, which is
    commonly used in regression tasks. It computes the average of the squared
    differences between predicted values and ground-truth targets.

    Example:
    --------
        mse = MeanSquareError()
        loss = mse(y_true, y_pred)

    Notes:
    ------
    This implementation wraps the input with `GradientReflector` if it's not already
    an instance, enabling automatic differentiation support for training.

    how to use : 

        from LittleLearn.DeepLearning import loss 

        loss_fn = loss.MeanSquareError()

        import numpy as np 

        y_true = np.array([1,2,3,4]).reshape(-1,1)

        y_pred = np.array([0.8,1.6,3.4,3.7])

        loss_fn(y_true,y_pred)
    
    Author : Candra Alpin Gunawan 
    """
    def __init__ (self) :
        pass
    
    def __call__ (self,y_true,y_pred) :
        """
        Computes the Mean Squared Error (MSE) loss between the true values and predicted values.

        Parameters
        ----------
        y_true : array-like
            The ground truth target values.
        y_pred : array-like or GradientReflector
            The predicted values from the model. If not an instance of GradientReflector,
            it will be automatically wrapped.

        Returns
        -------
        float
            The computed mean squared error loss.
        """
        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor
        
        return ll.GradientReflector.mse_loss(
            y_train=y_true,y_pred=y_pred
        )

class MAELoss :
    """
    MeanAbsoluteError (MAE)
    -----------------------
    This class implements the Mean Absolute Error (MAE) loss function,
    commonly used in regression tasks. MAE calculates the average of the absolute
    differences between predicted values and actual target values.

    Example:
    --------
        mae = MeanAbsoluteError()
        loss = mae(y_true, y_pred)

    Notes:
    ------
    This function wraps the prediction input with `GradientReflector` if it is not
    already an instance, enabling gradient tracking for optimization during training.

    Author : Candra Alpin Gunawan 
    """
    def __call__(self, y_true,y_pred):
        """
        Computes the Mean Absolute Error (MAE) loss between the true values and predicted values.

        Parameters
        ----------
        y_true : array-like
            The ground truth target values.
        y_pred : array-like or GradientReflector
            The predicted values from the model. If not an instance of GradientReflector,
            it will be automatically wrapped.

        Returns
        -------
        float
            The computed mean absolute error loss.
        """

        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        if isinstance(y_true,ll.Tensor) : 
            y_true = y_true.tensor
        
        return ll.GradientReflector.mae_loss(
            y_train=y_true,y_pred=y_pred
        )

class BinaryCrossentropy :
    """
    BinaryCrossentropy
    ------------------
    This class implements the Binary Crossentropy loss function, typically used in
    binary classification tasks. It measures the dissimilarity between the ground truth
    labels and predicted probabilities or logits.

    Parameters
    ----------
    from_logits : bool, default=False
        If True, `y_pred` is assumed to be logits (unnormalized scores) and will be passed
        through a sigmoid function internally. If False, `y_pred` is expected to be probabilities.
    
    epsilon : float, default=1e-6
        A small constant added for numerical stability during logarithmic computations,
        preventing log(0).

    Example:
    --------
        bce = BinaryCrossentropy(from_logits=True)
        loss = bce(y_true, y_pred)

    Notes:
    ------
    The predicted input will be wrapped with `GradientReflector` if it is not already an instance,
    allowing support for automatic differentiation during training.

    Author : Candra Alpin Gunawan 
    """

    def __init__(self,from_logits = False,epsilon=1e-6) :
        self.epsilon = epsilon 
        self.from_logits = from_logits

    def __call__ (self,y_true,y_pred) :
        """
        Computes the Binary Crossentropy loss between true labels and predictions.

        Parameters
        ----------
        y_true : array-like
            The ground truth binary labels (0 or 1).
        y_pred : array-like or GradientReflector
            The predicted values. Can be raw logits or probabilities, depending on `from_logits`.

        Returns
        -------
        float
            The computed binary crossentropy loss.
        """
        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
            
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor
        
        if self.from_logits :
            ll.GradientReflector.sigmoid(y_pred)
        
        return ll.GradientReflector.bce_loss(y_true,y_pred,
                                             epsilon=self.epsilon)

class HuberLoss :
    """
    HuberLoss
    ---------
    This class implements the Huber loss function, which is used in regression tasks
    as a robust alternative to Mean Squared Error (MSE). Huber loss is less sensitive 
    to outliers by combining MSE and Mean Absolute Error (MAE) based on a threshold `delta`.

    The loss is defined as:
        - 0.5 * (error)^2           if |error| <= delta
        - delta * (|error| - 0.5 * delta)   otherwise

    Parameters
    ----------
    delta : float, default=1.0
        The threshold at which the loss function transitions from quadratic to linear.

    Example:
    --------
        huber = HuberLoss(delta=1.0)
        loss = huber(y_true, y_pred)

    Notes:
    ------
    The `y_pred` input will be wrapped with `GradientReflector` if it is not already,
    to support automatic differentiation during training.

    Author : Candra Alpin Gunawan 
    """
    def __init__ (self,delta=1.0) :
        self.delta = delta 
    
    def __call__ (self,y_true,y_pred) :
        """
        Computes the Huber loss between true values and predicted values.

        Parameters
        ----------
        y_true : array-like
            The ground truth target values.
        
        y_pred : array-like or GradientReflector
            The predicted values from the model.

        Returns
        -------
        float
            The computed Huber loss.
        """
        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor 

        return ll.GradientReflector.huber_loss(y_true,y_pred,delta=self.delta)
    
class SparseCategoricalCrossentropy :
    """
    SparseCategoricalCrossentropy
    -----------------------------
    This class implements the Sparse Categorical Crossentropy loss function, used for
    multi-class classification problems where the target labels are provided as integers
    instead of one-hot encoded vectors.

    Parameters
    ----------
    from_logits : bool, default=False
        If True, `y_pred` is assumed to be raw logits and will be passed through a softmax function.
        If False, `y_pred` is expected to be a probability distribution.

    epsilon : float, default=1e-6
        A small constant added for numerical stability to avoid log(0) during computation.

    Example:
    --------
        scce = SparseCategoricalCrossentropy(from_logits=True)
        loss = scce(y_true, y_pred)

    Notes:
    ------
    The `y_pred` will be wrapped with `GradientReflector` if not already an instance.
    This enables support for gradient propagation during backpropagation.

    Author : Candra Alpin Gunawan 
    """

    def __init__(self,epsilon=1e-6) :
        self.epsilon = epsilon
    
    def __call__ (self,y_true,y_pred) :
        """
        Computes the Sparse Categorical Crossentropy loss between integer class labels and predictions.

        Parameters
        ----------
        y_true : array-like (integers)
            The ground truth labels as class indices (e.g., [1, 0, 3]).
        
        y_pred : array-like or GradientReflector
            The predicted logits or probability distributions from the model.

        Returns
        -------
        float
            The computed sparse categorical crossentropy loss.
        """
        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor
        
        return ll.GradientReflector.sparse_cross_entropy(
            y_true,y_pred
        )
    
class OneHotCrossEntropy :
    def __call__ (self,y_true,y_pred) :
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor
        
        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        return ll.GradientReflector.cross_entropy(y_true,y_pred)

class MaskedOneHotCrossEntropy :
    def __call__(self,y_true,y_pred,mask) :
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor
        
        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        return ll.GradientReflector.masked_cross_entropy(
            y_true,y_pred,mask=mask
        )
    
class MaskedSparseCrossEntropy :
    def __call__ (self,y_true,y_pred,mask) :

        if not isinstance(y_pred,ll.Tensor) :
            y_pred = ll.Tensor(y_pred)
        
        if isinstance(y_true,ll.Tensor) :
            y_true = y_true.tensor
        
        return ll.GradientReflector.masked_sparse_cross_entropy(y_true,y_pred,mask)


def bce_loss(y_true,y_pred,epsilon = 1e-5) :
    if not isinstance(y_pred,ll.Tensor) :
        y_pred = ll.Tensor(y_pred)
    
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor 
    
    return ll.GradientReflector.bce_loss(y_true,y_pred,epsilon=epsilon)

def mse_loss (y_true,y_pred) :
    if not isinstance(y_pred,ll.Tensor) :
        y_pred = ll.Tensor(y_pred)
    
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor
    
    return ll.GradientReflector.mse_loss(y_true,y_pred)

def mae_loss (y_true,y_pred) :
    if not isinstance(y_pred,ll.Tensor):
        y_pred =  ll.Tensor(y_pred)
    
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor 
    
    return ll.GradientReflector.mse_loss(y_true,y_pred)

def sparse_crossentropy (y_true,y_pred) :
    
    if not isinstance(y_pred,ll.Tensor) :
        y_pred = ll.Tensor(y_pred)
    
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor 
    
    return ll.GradientReflector.sparse_cross_entropy(y_true,y_pred)

def onehot_crossentropy (y_true,y_pred) :
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor
    
    if not isinstance(y_pred,ll.Tensor) :
        y_pred = ll.Tensor(y_pred)
    
    
    return ll.GradientReflector.cross_entropy(y_true,y_pred)

def masked_sparse_crossentropy (y_true,y_pred,mask) :
    if not isinstance(y_pred,ll.Tensor) :
        y_pred = ll.Tensor(y_pred)
    
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor 
    
    return ll.GradientReflector.masked_sparse_cross_entropy(y_true,y_pred,mask)

def masked_onehot_crossentropy (y_true,y_pred,mask) :
    if isinstance(y_true,ll.Tensor) :
        y_true = y_true.tensor
    
    if not isinstance(y_pred,ll.Tensor) :
        y_pred = ll.Tensor(y_pred)
        
    return ll.GradientReflector.masked_cross_entropy(
        y_true,y_pred,mask
    )