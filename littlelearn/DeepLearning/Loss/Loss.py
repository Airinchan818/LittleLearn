from typing import Literal
import littlelearn as ll 
class MeanSquareError :
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
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.meansquareerror(y_true)

class MeanAbsoluteError :
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
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.meanabsoluteerror(y_true)

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
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.binarycrossetnropy(y_true,epsilon=self.epsilon,from_logits=self.from_logits)

class CaterigocallCrossentropy :
    """
    CategoricalCrossentropy
    ------------------------
    This class implements the Categorical Crossentropy loss function, which is used
    in multi-class classification tasks where each sample belongs to one of several classes.

    Parameters
    ----------
    from_logits : bool, default=False
        If True, `y_pred` is expected to be raw logits and will be internally passed through
        a softmax function. If False, `y_pred` is expected to be a probability distribution.
    
    epsilon : float, default=1e-6
        A small constant for numerical stability to avoid taking the logarithm of zero.

    Example:
    --------
        cce = CategoricalCrossentropy(from_logits=True)
        loss = cce(y_true, y_pred)

    Notes:
    ------
    The input `y_pred` will be wrapped using `GradientReflector` if not already,
    to support gradient computation during backpropagation.

    Author : Candra Alpin Gunawan 
    """

    def __init__(self,from_logits=False,epsilon=1e-6) :
        self.epsilon = epsilon
        self.from_logits = from_logits

    def __call__ (self,y_true,y_pred) :
        """
        Computes the Categorical Crossentropy loss between the true labels and predictions.

        Parameters
        ----------
        y_true : array-like
            Ground truth one-hot encoded labels.
        
        y_pred : array-like or GradientReflector
            Model predictions. These can be logits or probabilities depending on `from_logits`.

        Returns
        -------
        float
            The computed categorical crossentropy loss.
        """
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.categoricallcrossentropy(y_true,self.epsilon,from_logits=self.from_logits)

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
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.hubber_loss(y_true,self.delta)

class SparseCategoricallCrossentropy :
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

    def __init__(self,from_logits=False,epsilon=1e-6) :
        self.epsilon = epsilon
        self.from_logits = from_logits
    
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
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.sparsecategoricallcrossentropy(y_true,self.epsilon,from_logits=self.from_logits)