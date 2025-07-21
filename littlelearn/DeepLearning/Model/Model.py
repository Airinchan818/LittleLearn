import numpy as np 
from typing import Literal 
import traceback
from littlelearn.DeepLearning import optimizers
import matplotlib.pyplot as plt 

class Sequential :
    """
    Sequential
    ----------

    the Sequential Layers Wrapper, this layers Wrapper work by list layers work,
    you just use it with : \n 
    model = Sequential([...])

    Sequential just suport on layers with one input like Dense,LSTM or other. 
    how its work: 

    Sequential just need apa parameters list layers:

    layers1 = dl.layers.Dense(32)
    layers2 = dl.layers.LayerNormalization()

    Sequential([layers1,layers2],name='Sequential model')

    Example:
    ---------
    from LittleLearn import DeepLearning as dl 

    model = dl.Model.Sequential([
        dl.layers.Dense(32,activation=dl.activations.Relu()),
        dl.layers.Dense(1,activation = 'sigmoid')
    ])

    model.build_model(optimizer = dl.optimizers.Adam(),loss=dl.loss.BinaryCrossentropy())

    Note:
    ----------
    AutoClipGradient on by default in Gradient Reflector backend, and can't set cause the gradient exploading problems
    when you need more control on your model, layers can you train with out Layers Wrapper. 
    you can get all control on your model. 
    
    Author:
    -------
    Candra Alpin Gunawan 
    """
    def __init__ (self,Layers=list(),name='sequential Model'):
        self.__layers = Layers
        self.__loss_record = list()
        self.gradient = None
        self.__optimizer = None 
        self.__loss = None 
        self.__build_status = False
        self.name = name 
        self.__weight_tmp = list()
        self.loss_val = None 
    
    def __call__(self,x):
        for layers in self.__layers :
            x  = layers(x)
        return x 
    
    def build_model(self,optimizer,loss) :
        """
        this is for build model 

        how to use  : 
        example : \n
        model = Sequential([....])\n
        optim_fn = Littlearn.Deeplearning.optimizers.Adam()\n
        loss_fn = Littlelearn.DeppLearning.loss.BinaryCrossEntropy()\n
        model.build_model(optimizer = optim_fn,loss = loss_fn)
        """
        self.__optimizer = optimizer
        self.__loss = loss
        self.__build_status  = True
    
    def detail(self) :
        """
            call it for look A model detail like Parameter ans size model

            example:
                model.detail()
        """ 

        param_size = 0 

        if self.loss_val is None  :
            print(f"Model Name : {self.name}")
            for layers in self.__layers :
                print(f"Layers_Name : {layers.name} \n output : unbuilt \n parameter size : unbuilt")
        else  :
            print(f"Model Name : {self.name}")
            for layers in self.__layers :
                print(f"Layers_Name : {layers.name} \n output : {layers.out_shape} \n parameter size : {layers.parameter}")
                param_size += layers.parameter
            print(f"Total Trained Parameter : {param_size}")
            param_size += self.__optimizer.parameter 
            print(f"Optimizers parameters : {self.__optimizer.parameter}")
            dtype = None 
            if self.__layers[-1].weight.dtype == np.float32 :
                dtype = 32 
            elif self.__layers[-1].weight.dtype == np.float64 :
                dtype = 64 
            elif self.__layers[-1].weight.dtype == np.float16 :
                dtype = 16 
            elif self.__layers[-1].weight.dtype  == np.float128 :
                dtype = 128 
            elif self.__layers[-1].weight.dtype  == np.float256 :
                dtype = 256 
            elif self.__layers[-1].weight.dtype  == np.float96 :
                dtype = 96 
            elif self.__layers[-1].weight.dtype  == np.float80 :
                dtype = 80 

            param_size = param_size * dtype 
            print(f"size_model = {param_size / (1024**2)}.mb")
    
    
    def fit(self,x,y,epochs:int,verbose:Literal[0,1] = 0) :
        """" 
        ##Remember !! \n :
        Sequential Models just suport by one input layers, its can't suport to use Multihead attention\n
        or another model that have more one input \n 
        #how to fit ? \n
            model.fit(x,y,epochs=10,verbose) \n 
            when : \n 
            x is your training datasets \n 
            y is your training target model \n 
            epochs is your iteration model training estimination 

        """
        try :
            if self.__build_status is False  :
                raise RuntimeError("The model must call build_model first")
            
            for epoch in range(epochs) :
                out = x 
                y_pred = self.__call__(out)
                self.loss_val = self.__loss(y,y_pred)
                self.__loss_record.append(self.loss_val.tensor)
                if verbose == 1 :
                    print(f"epoch {epoch + 1} || loss : {self.loss_val.tensor:.5f}")
                self.loss_val.AutoClipGradient()
                self.loss_val.backwardpass()
                self.__run_optimizer()
                self.loss_val.kill_grad()
                self.__weight_tmp.clear()
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def predict(self,x) :
        return self.__call__(x)

    def __run_optimizer(self):

            if isinstance(self.__optimizer,optimizers.GradientDescent) :
                RuntimeError("This Optimizers Not suport for Neural Network ")

            if isinstance(self.__optimizer,optimizers.Adam) :
                w = self.take_weight()
                self.__optimizer.apply_weight(w)
                self.__optimizer.forward_in_weight()

            if isinstance(self.__optimizer,optimizers.Rmsprop) :               
                w = self.take_weight()
                self.__optimizer.apply_weight(self.__weight_tmp)
                self.__optimizer.forward_in_weight()

            if isinstance(self.__optimizer,optimizers.Momentum) :             
                w = self.take_weight()
                self.__optimizer.apply_weight(self.__weight_tmp)
                self.__optimizer.forward_in_weight()
            
            if isinstance(self.__optimizer,optimizers.AdamW) :
                w = self.take_weight()
                self.__optimizer.apply_weight(w)
                self.__optimizer.forward_in_weight()
            
            if isinstance(self.__optimizer,optimizers.Adamax) :
                w = self.take_weight()
                self.__optimizer.apply_weight(w)
                self.__optimizer.forward_in_weight()
            
            if isinstance(self.__optimizer,optimizers.Lion) :
                w = self.take_weight()
                self.__optimizer.apply_weight(w)
                self.__optimizer.forward_in_weight()


    def take_weight(self):
        for layers in self.__layers:
            weight = layers.get_weight()
            if weight is not None:
                for w in weight:
                    self.__weight_tmp.append(w)
        return self.__weight_tmp 
    

    def plot_loss (self) :

        """
        its for see how many loss gonna down by epoch per epoch 
        """
        if len(self.__loss_record) <=0:
            raise RuntimeError("Model must fit first")
        loss = np.array(self.__loss_record)
        plt.title(self.name + " loss")
        plt.ylabel("loss values")
        plt.xlabel("epoch per step ")
        plt.plot(np.arange(len(loss)),loss)
        plt.show()
    
    def plot_graph_execution (self) :
        """
        this for you see what hapened in model, when you use Gradient Reflector
        """
        if self.loss_val is None :
            raise RuntimeError("Model must Fit first")
        self.loss_val.plot_trace_operation()


class AutoBuildModel : 

    """
    AutoBuildModel 
    -------------
        AutoBuildModel is API that can build Deep learning model by one line, just to 
        know what you model need and how deep level your model, you can get your model by 
        quickly time\n 

        Parameters:
        -----------
        num_target = int|None (optional) Default (None)
            its num target if model task for multiclassification even that is sentiment task
            but for sentiment task if num_target is None, will be Binary Classification as default. 
        
        type = Literal choice default(None) 
            for choice what sfesifik model you need by name in Literal choice. 
        
        level = Literal choice default(None)
            for choice deep level your model, its have 3 option (light,balance,deep)
        
        vocab_size : int default (None) Optional 
            this parameter just when you use AutoBuildModel with nlp task
        
        
    Note:
    --------
        AutoBuildModel work by layerWrapper backend, following by update library
        this Api can have new model. 

    Author:
    -------
    Candra Alpin Gunawan     
        


    """

    def __init__(self,num_target = None,vocab_size=None ,type:Literal['text-sentiment-lstm','mlp-regression',
        'mlp-classification','mlp-binaryclassification','text-sentiment-simple-rnn',
        'text-sentiment-gru'] = None , level : Literal ['light','balance','deep'] = None ) : 

        self.__type = type 
        self.__level = level 
        self.class_target = num_target
        self.__vocab = vocab_size

        self.__build_model() 

    
    def __build_model (self) : 
        from littlelearn.DeepLearning import layers,loss
        
        if self.__type not in ['text-sentiment-lstm','mlp-regression',
        'mlp-classification','mlp-binaryclassification','text-sentiment-simple-rnn',
        'text-sentiment-gru'] or None :
            raise RuntimeError(f"the type model not available for {self.__type}")
        
        if self.__level not in ['light','balance','deep'] or self.__level == None  :
            raise RuntimeError(f"the level model not available for {self.__level}")

        if self.__type == 'mlp-regression' :
            
            if self.__level == 'light' :
                self.__model = Sequential([
                    layers.Dense(32,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(1,activation='linear')
                ],name="light-Regression"),
                


            elif self.__level == 'balance' :
                self.__model = Sequential([
                    layers.Dense(32,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(1,activation='linear')
                ],name='balance-regression') 
            
            elif self.__level == 'deep' : 
                self.__model = Sequential([
                    layers.Dense(64,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(256,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(1,activation='linear')
                ],name='deep-regression')
            
            if isinstance(self.__model,Sequential) : 
                self.__model.build_model(
                    optimizer=optimizers.Adam(),
                    loss=loss.MeanSquareError()
                )

            
        if self.__type == 'mlp-binaryclassification' : 
            if self.__level == 'light' : 
                self.__model = Sequential([
                    layers.Dense(32,activation='relu'),
                    layers.Dense(32,activation='relu'),
                    layers.Dense(1,activation='sigmoid')
                ],name='light-binaryclassification')
            
            
            elif self.__level == 'balance' : 
                self.__model = Sequential([
                    layers.Dense(32,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(32,activation='relu'),
                    layers.Dense(1,activation='sigmoid')
                ],name='balance-binaryclassification')
            
            elif self.__level == 'deep' :
                self.__model = Sequential([
                    layers.Dense(64,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(256,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(1,activation='sigmoid')
                ],name='deep-binaryclassification')

            if isinstance(self.__model,Sequential) : 
                self.__model.build_model(
                    optimizer=optimizers.Adam(),
                    loss=loss.BinaryCrossentropy()
                )
                
        
        if self.__type == 'mlp-classification' :
            if self.class_target is None :
                raise RuntimeError("mlp-classification type need num target class for do task")
            
            if self.__level == 'light' : 
                self.__model = Sequential([
                    layers.Dense(32,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(self.class_target,activation='softmax')
                ],name='light-classification')

            elif self.__level == 'balance' : 
                self.__model = Sequential([
                    layers.Dense(64,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(self.class_target,activation='softmax')
                ],name='balance-mlpclassification')
            
            elif self.__level == 'deep' : 
                self.__model = Sequential([
                    layers.Dense(64,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(128,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(64,activation='relu'),
                    layers.Dense(32,activation='relu'),
                    layers.Dense(self.class_target,activation='softmax')
                ],name='deep-mlpclassification')
            
            if isinstance(self.__model,Sequential) : 
                self.__model.build_model(
                    optimizer=optimizers.Adam(),
                    loss=loss.SparseCategoricallCrossentropy()
                )
        
        if self.__type == 'text-sentiment-lstm' : 
            
            if self.__level == 'light' : 
                if self.class_target is None :
                    self.__model = Sequential([
                        layers.Embedding(input_dim=self.__vocab,output_dim=32),
                        layers.LSTM(32,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='light-lstm-sentiment')

                    if isinstance(self.__model,Sequential) : 
                        self.__model.build_model(
                            optimizer=optimizers.Rmsprop(),
                            loss=loss.BinaryCrossentropy()
                        )

                else :
                    self.__model = Sequential([
                        layers.Embedding(input_dim=self.__vocab,output_dim=32),
                        layers.LSTM(32,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='light-lstm-sentiment')
                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Rmsprop(),
                            loss=loss.SparseCategoricallCrossentropy()
                        )
        
            elif self.__level == 'balance' : 

                if self.class_target is None :
                    self.__model = Sequential([
                        layers.Embedding(input_dim=self.__vocab,output_dim=64),
                        layers.LSTM(64,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='balance-lstm-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Adam(),
                            loss=loss.BinaryCrossentropy())
                else : 
                    self.__model = Sequential([
                        layers.Embedding(input_dim=self.__vocab,output_dim=64),
                        layers.LSTM(64,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='balance-lstm-sentiment')
                    if isinstance(self.__model,Sequential) : 
                        self.__model.build_model(
                            optimizer=optimizers.Adam(),
                            loss = loss.SparseCategoricallCrossentropy()
                        )
            
            elif self.__level == 'deep' :
                if self.class_target is None :
                    self.__model = Sequential([
                        layers.Embedding(self.__vocab,128),
                        layers.LSTM(64,return_sequence=True),
                        layers.LSTM(32,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='deep-lstm-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.AdamW(),
                            loss=loss.BinaryCrossentropy()
                        )
                else :
                    self.__model = Sequential([
                        layers.Embedding(self.__vocab,128),
                        layers.LSTM(64,return_sequence=True),
                        layers.LSTM(32,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='deep-lstm-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.AdamW(),
                            loss = loss.SparseCategoricallCrossentropy()
                        )
        
        if self.__type == 'text-sentiment-simple-rnn' :
            
            if self.__level == 'light' :
                if self.class_target is None :
                    self.__model = Sequential([
                        layers.Embedding(self.__vocab,16),
                        layers.SimpleRNN(32),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='light-simplernn-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Rmsprop(),
                            loss=loss.BinaryCrossentropy()
                        )
                else  : 
                    self.__model = Sequential([
                        layers.Embedding(self.__vocab,32),
                        layers.SimpleRNN(64),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(self.class_target,activation='sigmoid')
                    ],name='light-simplernn-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Rmsprop(),
                            loss= loss.SparseCategoricallCrossentropy()
                        )
            
            elif self.__level == 'balance' :
                vocab = self.__vocab

                if self.class_target is None  :
                    self.__model = Sequential([
                        layers.Embedding(vocab,64),
                        layers.SimpleRNN(64),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='balance-simplernn-sentiment')
                    
                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Adam(),
                            loss = loss.BinaryCrossentropy()
                        )
                else : 
                    self.__model = Sequential([
                        layers.Embedding(vocab,64),
                        layers.SimpleRNN(64),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='balance-simplernn-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Adam(),
                            loss= loss.SparseCategoricallCrossentropy()
                        )
            
            elif self.__level == 'deep' :
                vocab = self.__vocab
                if self.class_target is None  :
                    self.__model = Sequential([
                        layers.Embedding(vocab,128),
                        layers.SimpleRNN(128),
                        layers.SimpleRNN(64),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='deep-simplernn-sentiment')

                    if isinstance(self.__model,Sequential) : 
                        self.__model.build_model(
                            optimizer=optimizers.AdamW(),
                            loss= loss.BinaryCrossentropy()
                        )
                
                else :
                    self.__model = Sequential([
                        layers.Embedding(vocab,128),
                        layers.SimpleRNN(128),
                        layers.SimpleRNN(64),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='deep-simplernn-sentiment')

                    if isinstance(self.__model,Sequential) : 
                        self.__model.build_model(
                            optimizer=optimizers.AdamW(),
                            loss = loss.SparseCategoricallCrossentropy()
                        )
            
        if self.__type == 'text-sentiment-gru' :
            vocab = self.__vocab

            if self.__level == 'light' : 
                if self.class_target is None : 
                    self.__model = Sequential([
                        layers.Embedding(vocab,32),
                        layers.GRU(32,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='light-gru-sentiment')
                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Rmsprop(),
                            loss = loss.BinaryCrossentropy()
                        )
                
                else :
                    
                    self.__model = Sequential([
                        layers.Embedding(vocab,32),
                        layers.GRU(32,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='light-gru-sentiment')

                    if isinstance(self.__model,Sequential) : 
                        self.__model.build_model(
                            optimizer= optimizers.Rmsprop(),
                            loss = loss.SparseCategoricallCrossentropy()
                        )
            
            elif self.__level == 'balance' :

                if self.class_target is None :

                    self.__model = Sequential([
                        layers.Embedding(vocab,64),
                        layers.GRU(64,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='balance-gru-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer= optimizers.Adam(),
                            loss = loss.BinaryCrossentropy()
                        )
                else :

                    self.__model = Sequential([
                        layers.Embedding(vocab,64),
                        layers.GRU(64,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='balance-gru-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.Adam(),
                            loss = loss.SparseCategoricallCrossentropy()
                        )
            
            elif self.__level == 'deep' :
                
                if self.class_target is None :
                    self.__model = Sequential([
                        layers.Embedding(vocab,128),
                        layers.GRU(128,return_sequence=True),
                        layers.GRU(64,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(1,activation='sigmoid')
                    ],name='deep-gru-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.AdamW(),
                            loss = loss.BinaryCrossentropy()
                        )
                else :

                    self.__model = Sequential([
                        layers.Embedding(vocab,128),
                        layers.GRU(128,return_sequence=True),
                        layers.GRU(64,return_sequence=True),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(64,activation='relu'),
                        layers.Dense(self.class_target,activation='softmax')
                    ],name='deep-gru-sentiment')

                    if isinstance(self.__model,Sequential) :
                        self.__model.build_model(
                            optimizer=optimizers.AdamW(),
                            loss = loss.SparseCategoricallCrossentropy()
                        )
    
    def fit(self,x,y,epochs:int,verbose : Literal[0,1] = 1) :
        if self.__vocab is None and self.__type in ['text-sentiment-simple-rnn',
        'text-sentiment-gru','text-sentiment-lstm'] : 
            raise RuntimeError("When use AutobuildModel for nlp task, vocab_size parameter must note None ")

        if self.__type == 'mlp-classification' and self.class_target is None :
            raise RuntimeError("'mlp-classification' asumtion you use multiclassification, so num_target can't be None ")
        if isinstance(self.__model,Sequential) :
            self.__model.fit(x,y,epochs=epochs,verbose=verbose)
    
    def predict(self,x) :
        
        if isinstance(self.__model,Sequential) :
            return self.__model(x)
    
    def take_weight (self,x) :
        
        if isinstance(self.__model,Sequential ) :
            return self.__model.take_weight()
    
    def plot_loss (self) :
        if isinstance(self.__model,Sequential) : 
            self.__model.plot_loss()
            
    def __call__(self,x):
        return self.__model(x)

    def detail(self) :
        if isinstance(self.__model,Sequential) :
            self.__model.detail()

    def plot_graph_execution(self) :
        if isinstance(self.__model,Sequential) : 
            self.__model.plot_graph_execution()