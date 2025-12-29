from typing import Literal
from littlelearn.DeepLearning import layers as la 
from littlelearn.DeepLearning import activations as activ
from littlelearn.DeepLearning import optimizers
from littlelearn import arange,expand_dims
import matplotlib.pyplot as plt 
import math 

class Trainer :
    """
        Trainer 
        --------------
        Trainer is Class Trainer spesially for Custom Model inheritrance by Component Class. 

        Parameter:
        ---------------
            Model: Component
                model object for training target
            datasets: Dataset
                custom datasets class that inheritance to Dataset class
        
        how to use:
        -------------------
            ```

                trainer = Trainer(model,datasets)
                trainer.build_model(Adam(),BinaryCrossentropy()) 
                model_trained = trainer.run(batch_size=32,verbose=1,epochs=10,shuffle=True)


            ```

        Author
        -----------------
        Candra Alpin Gunawan
    """
    def __init__ (self,Model,datasets):
        from littlelearn.DeepLearning.layers import Component
        from littlelearn.preprocessing import DataLoader,Dataset
   
        if not isinstance(Model,Component) :
            raise ValueError("Model must inheritance from Component class")
        if not isinstance(datasets,Dataset) :
            raise ValueError("datasets must class object inheretince from datasets class")
        
        self.__loader = DataLoader(datasets)
        self.model = Model 
        self.loss_hist = []
        self.optimizer = None 
        self.loss_fn = None 
        self.clipper = None 
    
    def build_model(self,optimizer ,loss_fn ) :
        """
            just call it for initialing optimizer and loss function 
            and when you need use clipper (gradient clipping) you can 
            fill clipper parameter by gradientclipper class 

        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn 


    
    def run (self,batch_size = 32,epochs = 1, verbose : Literal[0,1] = 0,shuffle : bool = False,
            device = "cpu") :
        """
            run Trainer for training model, use this function for training model with 
            Trainer

            parameter: \n 
                batch_size : int default = 32 
                    batch_size for split datasets
                
                epochs : int default =1
                    training loop range
                
                verbose : Literal [0,1] default = 0 
                    for showing mean total_loss per epoch
                
                shuffle : bool default = False 
                    for shuffling datasets while training run
                
                device : str default = "cou" 
                    datasets device, warning : datasets must 
                    converted to Tensor and in the same device 
                    with model
                
            output:
            trained model : Component



        """
        if verbose not in [0,1] :
            raise ValueError("Verbose just support by 0 or 1 ")
        
        if self.optimizer is None or self.loss_fn is None :
            raise ValueError("you not build_model() yet, please to call build_model() before run Trainer")
        
        from tqdm import tqdm 
        self.__loader.batch_size = batch_size
        self.__loader.shuffle = shuffle
        lost_hist = []
        for epoch in range(epochs) :
            total_loss = 0 
            iterator = tqdm(self.__loader)
            for x_train,y_train in iterator :
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                y_pred = self.model(x_train)
                loss = self.loss_fn(y_train,y_pred)
                loss.backwardpass()
                if isinstance(self.optimizer,optimizers.Optimizer) :
                    self.optimizer.step()
                
                loss.reset_grad()
                total_loss = total_loss + loss.tensor 
                iterator.set_description(f"epoch : {epoch + 1} / {epochs}")
                iterator.set_postfix(loss = loss.tensor)
            
            total_loss = total_loss / len(self.__loader)
            self.loss_hist.append(total_loss)
            if verbose == 1 :
                print(f"epoch : {epoch + 1} / {epochs} || global loss : {total_loss}")
    
    def return_model (self) :
        return self.model

                
    def plot_loss (self) :
        plt.title(f"{self.model.__class__.__name__} loss")
        plt.xlabel("epoch")
        plt.ylabel("values")
        plt.plot(self.loss_hist,color='red',label='loss')
        plt.legend()
        plt.grid(True)
        plt.show()

class MLPBlock(la.Component) :
    def __init__(self,dim : int) :
        super().__init__()
        self.linear = la.Linear(dim,dim)
        self.relu = activ.Relu()
    
    def forwardpass (self,x) :
        x = self.linear(x)
        return self.relu(x)

class MLP (la.Component) :
    def __init__ (self,input_dim : int, depth_dim : int,n_class : int,num_depth = 3,mode : Literal['reg','bin','class'] = None) :
        super().__init__()
        if mode is None or mode not in ['reg','bin','class']   :
            raise ValueError(f"{mode} not suported")
        self.mode = mode 
        self.linear1 = la.Linear(input_dim,depth_dim)
        self.seqmodel = la.Sequential([
            [la.Linear(depth_dim,depth_dim) for _ in range(num_depth)]
        ])
        self.fl = la.Linear(depth_dim,n_class)
    
    def forwardpass(self,x) :
        x = self.linear1(x)
        x = self.seqmodel(x)
        x = self.fl(x)
        if self.mode == 'bin' :
            activ.sigmoid(x)
        else :
            return x 

class SSM_Model (la.Component) :
    def __init__(self,vocab_size : int,ssm_dim : int,n_class : int,num_depth : int=2,mode : Literal["bin","class"]="class") :
        super().__init__()
        if mode is None or mode not in ['reg','bin','class']   :
            raise ValueError(f"{mode} not suported")
        self.mode = mode 
        self.embedding = la.Embedding(vocab_size=vocab_size,embedding_dim=ssm_dim)
        self.ssm = la.DiagonalSSM(ssm_dim)
        self.seq = la.Sequential([MLPBlock(ssm_dim * 2) for _ in range(num_depth)])
        self.fl = la.Linear(ssm_dim*2,n_class)
    
    def forwardpass(self,x) :
        x = self.embedding(x)
        x = self.ssm(x)
        x = x[:,-1,:]
        x = self.seq(x)
        x = self.fl(x)
        if self.mode == 'bin' :
            return activ.sigmoid(x)
        else :
            return x 

class RNN_Model (la.Component) :
    def __init__(self,vocab_size : int,rnn_dim : int,n_class : int,num_depth : int=2,mode : Literal["bin","class"]="class") :
        super().__init__()
        if mode is None or mode not in ['bin','class']   :
            raise ValueError(f"{mode} not suported")
        self.mode = mode 
        self.embedding = la.Embedding(vocab_size,rnn_dim)
        self.rnn = la.SimpleRNN(rnn_dim,return_sequence=True)
        self.seq  = la.Sequential(
           [ MLPBlock(rnn_dim*2) for _ in range(num_depth)]
        )
        self.fl = la.Linear(rnn_dim*2,n_class)
        self.pooling = la.GlobalAveragePooling1D()
    
    def forwardpass(self,x) :
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.pooling(x)
        x = self.seq(x)
        x = self.fl(x)
        if self.mode == 'bin' :
            return activ.sigmoid(x)
        else :
            return x 

class LSTM_Model (la.Component) :
    def __init__(self,vocab_size : int,lstm_dim : int,n_class : int,num_depth : int=2,mode : Literal["bin","class"]="class") :
        super().__init__()
        if mode is None or mode not in ['bin','class']   :
            raise ValueError(f"{mode} not suported")
        self.mode = mode 
        self.embedding = la.Embedding(vocab_size,lstm_dim)
        self.lstm = la.LSTM(lstm_dim)
        self.seq = la.Sequential([
            MLPBlock(lstm_dim *2 ) for _ in range(num_depth)
        ])
        self.fl = la.Linear(lstm_dim * 2,n_class)
        self.pooling = la.GlobalAveragePooling1D()
    
    def forwardpass(self,x) :
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.pooling(x)
        x = self.seq(x)
        x = self.fl(x)
        if self.mode == 'bin' :
            return activ.sigmoid(x)
        else :
            return x 

class MHATransformers (la.Component) :
    def __init__(
            self,vocab_size :int,embed_dim : int,
            ffn_dim : int,drop_rate : float,max_pos : int,num_depth : int = 3
            ,mode : Literal['Encoder','Decoder'] = 'Encoder',

    ) :
        super().__init__()
        if mode not in ['Encoder','Decoder'] :
            raise ValueError (f"{mode} not supported")

        self.block = la.Sequential([
        la.TransformersBlock(
            embed_dim=embed_dim,ffn_dim=ffn_dim,drop_rate=drop_rate,
            type='multi',mode=mode
            ) for _ in range(num_depth)
        ])
        self.scale = math.sqrt(float(embed_dim))
        self.embedding = la.Embedding(
            vocab_size=vocab_size,embedding_dim=embed_dim
        )
        self.pos_learn = la.Embedding(max_pos,embed_dim)
    
    def forwardpass(self,x) : 
        B,S = x.shape 
        x = self.embedding(x)
        x = x * self.scale
        pos  = arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = expand_dims(pos,axis=0)
        x = x + pos 
        x = self.block(x)
        return x 

class Transformers (la.Component) :
    def __init__(
            self,vocab_size :int,embed_dim : int,
            ffn_dim : int,drop_rate : int,max_pos : int,num_depth : int = 3
            ,mode : Literal['Encoder','Decoder'] = 'Encoder',

    ) :
        super().__init__()
        if mode not in ['Encoder','Decoder'] :
            raise ValueError (f"{mode} not supported")

        self.block = la.Sequential([
        la.TransformersBlock(
            embed_dim=embed_dim,ffn_dim=ffn_dim,drop_rate=drop_rate,
            type='single',mode=mode
            ) for _ in range(num_depth)
        ])
        self.scale = math.sqrt(float(embed_dim))
        self.embedding = la.Embedding(
            vocab_size=vocab_size,embedding_dim=embed_dim
        )
        self.pos_learn = la.Embedding(max_pos,embed_dim)
    
    def forwardpass(self,x) : 
        B,S = x.shape 
        x = self.embedding(x)
        x = x * self.scale
        pos  = arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = expand_dims(pos,axis=0)
        x = x + pos 
        x = self.block(x)
        return x 

class LatentConnectedModel(la.Component) :
    def __init__(self,
                 vocab_size : int,embed_dim : int,drop_rate : float,
                 max_pos : int,
                 num_depth : int = 3
                 ) :
        self.scale = math.sqrt(float(embed_dim))
        self.block = la.Sequential([
            la.LCMBlock(embed_dim,drop_rate) for _ in range(num_depth)]
        )
        self.pos_learn = la.Embedding(max_pos,embed_dim)
        self.embedding = la.Embedding(vocab_size,embed_dim)
    
    def forwardpass(self,x) :
        B,S = x.shape 
        x = self.embedding(x)
        x = x * self.scale
        pos = arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = expand_dims(pos,axis=0)
        x = x+pos 
        x = self.block(x)
        return x 

class LatentConnectedTransformers (la.Component) :
    def __init__(
            self,vocab_size :int,embed_dim : int,
            num_head : int,drop_rate : int,max_pos : int,num_depth : int = 3
            ,mode : Literal['Encoder','Decoder'] = 'Encoder',

    ) :
        super().__init__()
        if mode not in ['Encoder','Decoder'] :
            raise ValueError (f"{mode} not supported")
        
        if mode == 'Encoder' :
            self.block = la.Sequential([
                la.LCTBlock(
                    embed_dim=embed_dim,num_head=num_depth,
                    drop_rate=drop_rate,use_causal_mask=False
                ) for _ in range(num_depth)
            ])
        else :
            self.block = la.Sequential([
                la.LCTBlock(
                    embed_dim=embed_dim,num_head=num_depth,
                    drop_rate=drop_rate,use_causal_mask=True
                ) for _ in range(num_depth)
                ])
        
        self.scale = math.sqrt(float(embed_dim))
        self.embedding = la.Embedding(vocab_size,embed_dim)
        self.pos_learn = la.Embedding(max_pos,embed_dim)
    
    def forwardpass(self,x) :
        B,S = x.shape
        x = self.embedding(x)
        x= x * self.scale
        pos = arange(0,S,device=x.device)
        pos = self.pos_learn(x)
        x = expand_dims(x)
        x = x + pos 
        x = self.block(x)
        return x 