from typing import Literal
from littlelearn.DeepLearning import layers as la 
from littlelearn.DeepLearning import activations as activ
from littlelearn.DeepLearning import optimizers
import littlelearn as ll
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
            device = "cpu",dtype=(ll.float32,ll.float32)) :
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
        for epoch in range(epochs) :
            total_loss = 0 
            iterator = tqdm(self.__loader)
            for x_train,y_train in iterator :
                x_train = ll.to_tensor(x_train,dtype=dtype[0],device=device,requires_grad=False)
                y_train = ll.to_tensor(y_train,dtype=dtype[1],device=device,requires_grad=False)
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
            if verbose == 1 :
                print(f"epoch : {epoch + 1} / {epochs} || global loss : {total_loss}")
    
    def return_model (self) :
        return self.model
                

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
            ,mode : Literal['Encoder','Decoder'] = 'Encoder'

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
        pos  = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = ll.expand_dims(pos,axis=0)
        x = x + pos 
        x = self.block(x)
        return x 

class Transformers (la.Component) :
    def __init__(
            self,vocab_size :int,embed_dim : int,
            ffn_dim : int,drop_rate : int,max_pos : int,num_depth : int = 3
            ,mode : Literal['Encoder','Decoder'] = 'Encoder'

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
        pos  = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = ll.expand_dims(pos,axis=0)
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
        pos = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = ll.expand_dims(pos,axis=0)
        x = x+pos 
        x = self.block(x)
        return x 

class LatentConnectedTransformers (la.Component) :
    def __init__(
            self,vocab_size :int,embed_dim : int,
            num_head : int,drop_rate : int,max_pos : int,num_depth : int = 3
            ,mode : Literal['Encoder','Decoder'] = 'Encoder'

    ) :
        super().__init__()
        if mode not in ['Encoder','Decoder'] :
            raise ValueError (f"{mode} not supported")
        
        if mode == 'Encoder' :
            self.block = la.Sequential([
                la.LCTBlock(
                    embed_dim=embed_dim,num_head=num_head,
                    drop_rate=drop_rate,use_causal_mask=False
                ) for _ in range(num_depth)
            ])
        else :
            self.block = la.Sequential([
                la.LCTBlock(
                    embed_dim=embed_dim,num_head=num_head,
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
        pos = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(x)
        x = ll.expand_dims(x)
        x = x + pos 
        x = self.block(x)
        return x 
    
class BlockLinearAttn (la.Component) :
    def __init__(self,embed_dim,drop_rate,epsilon):
        super().__init__()
        self.attntion = la.LinearAttention(epsilon=epsilon)
        self.dropout1 = la.Dropout(drop_rate)
        self.dropout2 = la.Dropout(drop_rate)
        self.norm1 = la.RMSNorm(embed_dim,epsilon)
        self.norm2 = la.RMSNorm(embed_dim,epsilon)
        self.ffn = la.FeedForwardNetwork(embed_dim,embed_dim*4)
        
    def forwardpass(self,x) :
        norm = self.norm1(x)
        attn = self.attntion(norm,norm,norm)
        attn = self.dropout1(attn)
        x = x + attn 

        norm = self.norm2(x)
        ffn = self.ffn(norm)
        ffn = self.dropout2(ffn)
        x = x + ffn 
        
        return x 
    

class LinearAttentionEncoder (la.Component):
    def __init__ (self,vocab_size : int ,embed_dim : int ,drop_rate : float,num_depth:int,
                  maxpos : int,epsilon : float) :
        super().__init__()
        self.embedding = la.Embedding(vocab_size=vocab_size,
                                      embedding_dim=embed_dim)
        self.pos_learn = la.Embedding(maxpos,embed_dim)
        self.scale = vocab_size ** 0.5 
        self.block = la.Sequential(
            [
                BlockLinearAttn(embed_dim=embed_dim,drop_rate=drop_rate,epsilon=epsilon) for _ in range(num_depth)
            ]
        )
    
    def forwardpass(self,x) :
        _,S = x.shape 
        x = self.embedding(x)
        x = x * self.scale
        pos = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = ll.unsquezze(pos,axis=0)
        x = x + pos 
        x = self.block(x)
        return x

class MultiQATTNBlock (la.Component) :
    def __init__ (self,embed_dim,drop_rate,use_causalmask) :
        super().__init__()
        num_head = embed_dim // 64 
        if embed_dim < 64 :
            num_head = 2 
            
        self.attention = la.MultiQueryAttention(
            embed_dim=embed_dim,num_head=num_head,use_causalmask=use_causalmask
        )
        self.dropout1 = la.Dropout(drop_rate)
        self.norm = la.RMSNorm(embed_dim=embed_dim)
        self.norm2 = la.RMSNorm(embed_dim=embed_dim)
        self.dropout2 = la.Dropout(drop_rate)
        self.ffn = la.FeedForwardNetwork(embed_dim=embed_dim,ffn_dim=embed_dim*4)
    
    def forwardpass (self,x) :
        norm = self.norm(x)
        attn = self.attention(norm,norm,norm)
        attn = self.dropout1(attn)
        x = x + attn 

        norm = self.norm2(x)
        ffn = self.ffn(norm)
        ffn = self.dropout2(ffn)
        x = x + ffn 

        return x 


class MultiQueryAttentionTransformers (la.Component) :
    def __init__ (self,vocab_size : int ,embed_dim: int ,drop_rate : float,
                  num_depth : int,maxpos : int,mode : Literal['Encoder','Decoder'] = 'Encoder') :
        super().__init__()
        if mode not in ['Encoder','Decoder'] :
            raise ValueError(f"{mode} not supported")
        self.embedding = la.Embedding(vocab_size,embed_dim)
        self.pos_learn = la.Embedding(maxpos,embed_dim)
        if mode == 'Encoder':
            self.block = la.Sequential([
                MultiQATTNBlock(embed_dim,drop_rate,use_causalmask=False) for _ in range(num_depth) 
            ]) 
        else :
            self.block = la.Sequential([
                MultiQATTNBlock(embed_dim,drop_rate,use_causalmask=True) for _ in range(num_depth) 
            ]) 
        
        self.scale = embed_dim ** 0.5 
    
    def forwardpass (self,x) :
        _,S = x.shape
        x = self.embedding(x)
        x = x * self.scale
        pos = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = ll.unsquezze(pos,axis=0)
        x = x + pos 
        x = self.block(x)

        return x 

class TalkingHeadBlock (la.Component) :
    def __init__(self,embed_dim,drop_rate,use_causalmask) :
        super().__init__()
        num_head =embed_dim // 64 
        if embed_dim < 64 :
            num_head = 2 
        self.attention  = la.TalkingHeadAttention(
            embed_dim=embed_dim,num_heads=num_head,
            use_causalmask=use_causalmask
        )
        self.dropout1 = la.Dropout(drop_rate)
        self.dropout2 = la.Dropout(drop_rate)
        self.norm1 = la.RMSNorm(embed_dim)
        self.norm2 = la.RMSNorm(embed_dim)
        self.ffn = la.FeedForwardNetwork(
            embed_dim,embed_dim*4
        )
    
    def forwardpass(self,x) :
        norm = self.norm1(x)
        attn = self.attention(norm,norm,norm)
        attn = self.dropout1(attn)
        x = x + attn 

        norm = self.norm2(x)
        ffn = self.ffn(norm)
        ffn = self.dropout2(ffn)

        x = x + ffn 

        return x 

class TalkingHeadAttentionTransformers (la.Component) :
    def __init__ (self,vocab_size : int ,embed_dim : int ,num_depth : int,
                  drop_rate : float,maxpos : int,mode : Literal['Encoder','Decoder'] = 'Encoder') :
        super().__init__()
        if mode not in ['Encoder','Decoder'] :
            raise ValueError(f"{mode} is not supported")
        
        self.embedding = la.Embedding(vocab_size,embed_dim)
        self.pos_learn = la.Embedding(maxpos,embed_dim)

        if mode == 'Encoder' :
            self.block = la.Sequential([
                TalkingHeadBlock(embed_dim,drop_rate,use_causalmask=False) for _ in range(num_depth)
            ])
        else :
            self.block = la.Sequential([
                TalkingHeadBlock(embed_dim,drop_rate,use_causalmask=True) for _ in range(num_depth)
            ])
        
        self.scale = embed_dim ** 0.5
    
    def forwardpass(self,x) :
        _,S = x.shape 
        x = self.embedding(x)
        x = x * self.scale
        pos = ll.arange(0,S,device=x.device)
        pos = self.pos_learn(pos)
        pos = ll.unsquezze(pos,axis=0)
        x = x + pos 
        x = self.block(x)
        return x 

