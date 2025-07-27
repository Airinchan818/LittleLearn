import numpy as np 
from typing import Literal
import traceback
from littlelearn import DeepLearning 
import littlelearn  as ll 

class LinearRegression :
    """
    LinearRegression
    ----------------
    A simple linear regression model supporting customizable optimizers and loss functions.
    This model fits a linear relationship between input features and target values using
    gradient descent or a user-defined optimizer.

    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        The learning rate used for gradient descent updates.

    optimizer : callabel default Rmsprop
        optimizer is modular tools that run in Node,you have to fill optimizers class at this parameter.

    loss : callable default MeanSquaredError 
        A custom loss function. If not provided, mean squared error (MSE) will be used.

    Author : Candra Alpin Gunawan 
    """
    def __init__ (self,
                  optimizer = DeepLearning.optimizers.Rmsprop() ,
                  loss = DeepLearning.loss.MeanSquareError() ) :
        self.optimizer = optimizer
        self.__record_loss = list()
        self.Weight = None 
        self.bias = None 
        self.loss = loss 
        self.optimizer = optimizer
        self.__los_values = None 
        self.weight = []
        self.flag_train = False 
    
    def __build_Models (self,features) : 
        self.Weight = np.random.normal(0,scale=(2 / np.sqrt(features + features)),size=(features,1))
        self.bias = np.zeros((1,1))
        
        if not isinstance(self.Weight, ll.GradientReflector) :
            self.Weight = ll.convert_to_tensor(self.Weight)
        
        if not isinstance(self.bias,ll.GradientReflector) : 
            self.bias = ll.convert_to_tensor(self.bias)
        
        self.weight.append(self.Weight)
        self.bias 

    def fit(self,X,Y,Verbose : Literal[0,1] = 1,epochs=100) : 
        """
        Trains the linear regression model on the provided dataset.

        Parameters
        ----------
        X : ndarray
            Input features of shape (n_samples, n_features).

        Y : ndarray
            Target values of shape (n_samples, 1) or (n_samples,).

        Verbose : {0, 1}, optional (default=1)
            If 1, prints the loss at each epoch. If 0, disables logging.

        epochs : int, optional (default=100)
            The number of iterations over the training data.

        Raises
        ------
        ValueError
            If input dimensions are incorrect or Verbose is not 0 or 1.

        Author : Candra Alpin Gunawan 
        
        """
        if self.Weight is None or self.bias is None :
            self.__build_Models(X.shape[1])

        if not isinstance(X,ll.GradientReflector) : 
            X = ll.convert_to_tensor(X)
        
        self.flag_train = True 

        for epoch in range(epochs) :
            if len(X.shape) != 2 :
                print(f"Warning :: this X shape is = {X.shape}.X input must 2 dimentional do X.rehape(-1,1) before train")
                break
            y_pred = self.__call__(X)
            if isinstance(self.loss,DeepLearning.loss.MeanSquareError) : 
                self.__los_values = self.loss(Y,y_pred)
            
            if isinstance(self.loss,DeepLearning.loss.MeanAbsoluteError) : 
                self.__los_values = self.loss(Y,y_pred)
            
            if isinstance(self.loss, DeepLearning.loss.HuberLoss) :
                self.__los_values = self.loss(Y,y_pred)

            self.__los_values.AutoClipGradient()

            self.__los_values.backwardpass()

            if isinstance(self.optimizer,DeepLearning.optimizers.Adam) : 
                self.optimizer.apply_weight(self.weight) 
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Adamax) : 
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.AdamW) :
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Momentum) :
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Rmsprop) : 
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Lion) : 
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()

            self.__los_values.kill_grad()

            if Verbose == 1 :
                print(f"epoch : {epoch + 1} || loss : {self.__los_values.tensor}")
            
            self.__record_loss.append(self.__los_values.tensor)
                

            
    @property 
    def get_loss_record (self) :
        try :
            if len(self.__record_loss) == 0:
                raise ValueError("Model still not trained")
            return np.array(self.__record_loss)
        except Exception as e :
            e.add_note("You must Training model first")
            traceback.print_exception(type(e),e,e.__traceback__)

    
    def __call__ (self,X) :
        if self.Weight is None or self.bias is None :
            raise ValueError(f"None of Weight and bias can't do prediction")

        out = ll.matmul(X,self.Weight) + self.bias 
        return out 

    def plot_operation (self) : 
        """
            plot eporation Runtime is Gradient Reflector
        """
        if self.flag_train is False : 
            raise RuntimeError("fit model fisrt")
        self.__los_values.plot_trace_operation()

class LogisticRegression:
    """
    LogisticRegression
    ------------------
    A simple binary logistic regression model that supports gradient descent or a custom optimizer
    for training. Uses the sigmoid function to produce probability outputs and cross-entropy loss
    for optimization.

    Parameters
    ----------
    learning_rate : float, optional (default=0.001)
        The step size for parameter updates during training.

    optimizer : callabel default Rmsprop
        optimizer is modular tools that run in Node,you have to fill optimizers class at this parameter.

    epsilon : float, optional (default=1e-5)
        A small value added to the log function to prevent numerical instability (e.g., log(0)).
    
    Author : Candra Alpin Gunawan 
    """

    def __init__(self, optimizer=DeepLearning.optimizers.Rmsprop(), epsilon=1e-5):
        self.optimizer = optimizer
        self.Weight = None
        self.bias = None
        self.__record_loss = []
        self.__record_accuracy = []
        self.epsilon = epsilon
        self.__loss_val = None 
        self.__loss = DeepLearning.loss.BinaryCrossentropy()
        self.weight = []
        self.flag_train = False 

    def __build_Models(self, features):

        self.Weight = np.random.normal(0, scale=(2 / np.sqrt(features + features)), size=(features, 1))
        self.bias = np.zeros((1, 1))

        if not isinstance(self.Weight , ll.GradientReflector) :
            self.Weight = ll.convert_to_tensor(self.Weight)
        
        if not isinstance(self.bias,ll.GradientReflector) : 
            self.bias = ll.convert_to_tensor(self.bias)
        
        self.weight.append(self.Weight)
        self.weight.append(self.bias)

        

    def fit(self, X, Y, epochs=100, verbose: Literal[0, 1] = 1):
        """
        Trains the logistic regression model using binary cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        Y : np.ndarray
            Target labels of shape (n_samples, 1). Must be binary (0 or 1).

        epochs : int, optional (default=100)
            Number of training iterations.

        verbose : {0, 1}, optional (default=1)
            If 1, displays training loss and accuracy after each epoch.

        Raises
        ------
        ValueError
            If input shapes are invalid or verbose is not 0 or 1.
        """
        self.flag_train = True 
        if self.Weight is None or self.bias is None:
            self.__build_Models(X.shape[-1])
        
        if not isinstance(X,ll.GradientReflector) : 
            X = ll.GradientReflector(X)
        
        for epoch in range(epochs) : 
            y_pred = self.__call__(X)

            if isinstance(self.__loss,DeepLearning.loss.BinaryCrossentropy) :
                self.__loss_val = self.__loss(Y,y_pred)
            else : 
                raise RuntimeError("loss just suport by BinaryCrossentropy")
            
            self.__loss_val.AutoClipGradient()
            self.__loss_val.backwardpass()
            if isinstance(self.optimizer,DeepLearning.optimizers.Adam) : 
                self.optimizer.apply_weight(self.weight) 
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Adamax) : 
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.AdamW) :
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Momentum) :
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Rmsprop) : 
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            if isinstance(self.optimizer,DeepLearning.optimizers.Lion) : 
                self.optimizer.apply_weight(self.weight)
                self.optimizer.forward_in_weight()
            
            self.__loss_val.kill_grad()

            accuracy = (y_pred.tensor > 0.5).astype(int) == Y
            self.__record_accuracy.append(accuracy)
            self.__record_loss.append(self.__loss_val.tensor)
            
            if verbose == 1 :
                print(f"epoch : {epoch + 1} || loss : {self.__loss_val.tensor} || Accuracy : {np.mean(accuracy)}")

            
            
            

    def __call__(self, X):
        """
        Makes predictions using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted probabilities (between 0 and 1).

        Raises
        ------
        ValueError
            If weights or bias are not initialized.
        """
    
        if self.Weight is None or self.bias is None:
            raise ValueError("Fit Models First.")
        score = ll.matmul(X, self.Weight) + self.bias
        out = DeepLearning.activations.sigmoid(score)
        return out 

    @property
    def get_loss_record(self):
        try:
            if len(self.__record_loss) == 0:
                raise ValueError("Model has not been trained.")
            return np.array(self.__record_loss)
        except Exception as e:
            e.add_note("Train the model using `.fit()` before accessing loss history.")
            traceback.print_exception(type(e), e, e.__traceback__)

    @property
    def get_accuracy_record(self):
        try:
            if len(self.__record_loss) == 0:
                raise ValueError("Model has not been trained.")
            return np.array(self.__record_accuracy)
        except Exception as e:
            e.add_note("Train the model using `.fit()` before accessing accuracy history.")
            traceback.print_exception(type(e), e, e.__traceback__)
    
    def plot_operation(self) : 
        """
            plot eporation Runtime is Gradient Reflector
        """
        if self.flag_train is False  :
            raise RuntimeError("Fit model first")
        self.__loss_val.plot_trace_operation()
