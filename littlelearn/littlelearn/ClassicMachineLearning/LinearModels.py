import numpy as np 
from typing import Literal
import traceback

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

    optimizer : callable or None, optional
        A custom optimizer function. If provided, it should accept weights, bias,
        their gradients, and optionally the epoch, and return updated weights and bias
        as a dictionary with keys `'weight'` and `'bias'`.

    loss : callable or None, optional
        A custom loss function. If not provided, mean squared error (MSE) will be used.

    Author : Candra Alpin Gunawan 
    """
    def __init__ (self,learning_rate = 0.01,
                  optimizer = None ,
                  loss = None ) :
        self.optimizer = optimizer
        self.learning_rate = learning_rate 
        self.__record_loss = list()
        self.Weight = None 
        self.bias = None 
        self.loss = loss 
        self.optimizer = optimizer
    
    def __build_Models (self,features) : 
        self.Weight = np.random.normal(0,scale=(2 / np.sqrt(features + features)),size=(features,1))
        self.bias = np.zeros((1,features))

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
        for epoch in range(epochs) :
            if len(X.shape) != 2 :
                print(f"Warning :: this X shape is = {X.shape}.X input must 2 dimentional do X.rehape(-1,1) before train")
                break
            y_pred = np.dot(X,self.Weight) + self.bias
            if self.loss is None :
                loss = np.mean(np.power((y_pred - Y),2))
            else :
                loss = self.loss(Y,y_pred)
            if self.optimizer is None :
                if len(Y.shape) != 2 :
                    print(f"Warning : Y shape is : {Y.shape} is Not Compatible You must do reshape to Y.reshape(-1,1)")
                    break
                else :
                    gradient_w = (-2/len(Y)) * np.dot(X.T,(Y - y_pred))
                    gradient_b = (-2/len(Y)) * np.sum(Y - y_pred)
                    self.Weight -= self.learning_rate * gradient_w
                    self.bias -= self.learning_rate * gradient_b
            elif self.optimizers is not None :
                try :
                    gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                    gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                    grad = self.optimizers(self.Weight,self.bias,gradient_w,gradient_b,epoch)
                    self.Weight = grad['weight']
                    self.bias = grad['bias']
                except :
                    gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                    gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                    grad = self.optimizers(self.Weight,self.bias,gradient_w,gradient_b)
                    self.Weight = grad['weight']
                    self.bias = grad['bias']
            if Verbose != 1 or Verbose != 0:
                try :
                    raise ValueError("Verbose Values is not Valid")
                except Exception as e :   
                    e.add_note("Error : Verbose just choice [0 / 1] and you input {Verbose}".format(Verbose))
                    traceback.print_exception(type(e),e,e.__traceback__)
            if Verbose ==  1 :
                print(f"epoch : {epoch} || Loss : {loss:.6f}")
            self.__record_loss.append(loss)
            
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
        try :
            if self.Weight is None or self.bias in None :
                ValueError(f"None of Weight and bias can't do prediction")
            return np.dot(X,self.Weight) + self.bias
            
        except Exception as e :
            if self.Weight is None or self.bias is None : 
                e.add_note("Error : You must do Model.build_Models(features) at your model")
                e.add_note(f"Detail : Weight : {self.Weight} bias : {self.bias}")
                traceback.print_exception(type(e),e,e.__traceback__)
                raise

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

    optimizer : callable or None, optional
        A custom optimizer function that updates weights and bias. If provided, it must return a dictionary 
        with keys 'weight' and 'bias'.

    epsilon : float, optional (default=1e-5)
        A small value added to the log function to prevent numerical instability (e.g., log(0)).
    
    Author : Candra Alpin Gunawan 
    """

    def __init__(self, learning_rate=0.001, optimizer=None, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.optimizers = optimizer
        self.Weight = None
        self.bias = None
        self.__record_loss = []
        self.__record_accuracy = []
        self.epsilon = epsilon

    def __build_Models(self, features):

        self.Weight = np.random.normal(0, scale=(2 / np.sqrt(features + features)), size=(features, 1))
        self.bias = np.zeros((1, features))

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
        if self.Weight is None or self.bias is None:
            self.__build_Models(X.shape[1])

        for epoch in range(epochs):
            if len(Y.shape) != 2 or len(X.shape) < 2:
                print("Warning: X or Y must be 2D. Use X.reshape(-1, 1) or Y.reshape(-1, 1) before training.")
                break
            elif len(X.shape) > 2:
                print("Warning: Model only supports 2D input data.")
                break


            scores = np.dot(X, self.Weight) + self.bias
            y_pred = 1 / (1 + np.exp(-scores))


            loss = (-1 / len(Y)) * np.sum(Y * np.log(y_pred + self.epsilon) + (1 - Y) * np.log(1 - y_pred + self.epsilon))

            accuracy = np.mean((y_pred > 0.5).astype(int) == Y)

            gradient_w = (1 / len(Y)) * np.dot(X.T, (y_pred - Y))
            gradient_b = (1 / len(Y)) * np.sum(y_pred - Y)

            if self.optimizers is None:
                self.Weight -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b
            else:
                try:
                    grad = self.optimizers(self.Weight, self.bias, gradient_w, gradient_b, epoch)
                except:
                    grad = self.optimizers(self.Weight, self.bias, gradient_w, gradient_b)
                self.Weight = grad['weight']
                self.bias = grad['bias']

            if verbose not in [0, 1]:
                try:
                    raise ValueError("Invalid value for 'verbose'. Expected 0 or 1.")
                except Exception as e:
                    e.add_note("Use 0 to disable logging, or 1 to enable logging.")
                    traceback.print_exception(type(e), e, e.__traceback__)
                    raise

            if verbose == 1:
                print(f"Epoch: {epoch} || Loss: {loss:.6f} || Accuracy: {accuracy:.6f}")

            self.__record_loss.append(loss)
            self.__record_accuracy.append(accuracy)

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
        try:
            if self.Weight is None or self.bias is None:
                raise ValueError("Model is not initialized.")
            score = np.dot(X, self.Weight) + self.bias
            return 1 / (1 + np.exp(-score))
        except Exception as e:
            e.add_note(f"Weight = {self.Weight}, Bias = {self.bias}")
            e.add_note("Use `build_Models(features)` before making predictions.")
            traceback.print_exception(type(e), e, e.__traceback__)
            raise

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
