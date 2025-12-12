from littlelearn.DeepLearning import layers
import littlelearn as ll 
import numpy as np 
from typing import Literal

class AutoTransformers :
    """
        # AutoTransformers 
            this is a Transformers builder by one line, for fast prototype or light 
            transformers task. AutoTransformers give transformers Model Building just 
            one step on this API. 
            but for flexiblelity model must learn by Custom mode, because for add another layers
            or spesific task. 
        
        Parameter 
        --------------
            int d_model 
                like units in layers, its asumtion your model size at dimention Model
            
            int vocab_size 
                how many vocabulary at your datasets, this for embedding layers that can knowing
                how many need matriks lookup building at weight
            
            int ffn_size 
                feed forward units size
            
            int maxpos 
                how many maximum positional for every input or your datasets 
            
            optional type 
                your Transformers type like Encoder or Decoder and another type 
            
            optional Head_type 
                for choice MultiHead or Single Head at Attention Mechanism 
            
            optional level 
                how deep model at level 
            
            optional PosEncoding
                learn for Positional Embedding and constant for PositionalEncoding Sinusoidal
        
        example 
        ------------
                from littlelearn.DeepLearning.Model import AutoTransformers
                
                model = AutoTransformers(
                    d_model = 128,vocab_size = 100000,
                    ffn_size = 512,maxpos=200,type='decoder-nlp',
                    Head_type='Multi',level='balance',
                    PosEncoding='learn'
                )
        
        Author: Candra Alpin Gunawan 
    """

    def __init__ (self,d_model,vocab_size,ffn_size,drop_rate : float = 0.1
                  ,maxpos : int = 100 ,type:Literal['encoder-nlp','decoder-nlp','decoder-cross'] = 'encoder-nlp' ,
                  Head_type : Literal['Multi','Single'] = 'Multi'
                  ,level : Literal['light','balance','deep'] = 'light',
                  PosEncoding : Literal['learn','constant'] = 'learn',
                  NormMode : Literal ['postnorm','prenorm'] = 'prenorm') :
        self.type = type 
        self.level = level 
        self.Head_type = Head_type
        self.ffn_size = ffn_size
        self.maxpos = maxpos 
        self.vocab_size = vocab_size
        self.d_model = d_model 
        self.posencoding_type = PosEncoding
        self.NormMode = NormMode
        self.drop_rate = drop_rate
        self.__build_model()
    
    class __BasicTransformers :
        def __init__ (self):
            self.signal_type = None
            self.d_model = None 
            self.maxpos = None 
            self.Embedding = None 

        
        def get_weight(self) :
            NotImplementedError()
        

    class __Encoder_MHA_3block (__BasicTransformers) :
        def __init__(self,d_model,vocab_size,ffn,max_pos,signal_type,drop_rate :float = 0.1 ,
                     normMode : Literal['prenorm','postnorm']='prenorm') :
            self.signal_type = signal_type
            self.d_model = d_model 
            self.drop_rate = drop_rate
            self.maxpos = max_pos
            self.Embedding = layers.Embedding(vocab_size,d_model)
            self.block1 = layers.BlockEncoder_MHA(num_head=2,d_model=d_model,ffn=ffn,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockEncoder_MHA(num_head=2,d_model=d_model,ffn = ffn,NormMode=normMode,drop_rate=self.drop_rate )
            self.block3 = layers.BlockEncoder_MHA(num_head=2,d_model=d_model,ffn=ffn,NormMode=normMode,drop_rate=self.drop_rate)
            if self.signal_type == 'learn' :
                self.Positional_encoding = layers.Embedding(input_dim=max_pos,output_dim=d_model)
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.Positional_encoding
                ]            
            else : 
                self.Positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(maxpos=max_pos,d_model=d_model)
                self.Positional_encoding = ll.GradientReflector(self.Positional_encoding,_op='Positional')
                self.__node_layers = [
                    self.Embedding,self.block1,
                    self.block2,self.block3
                ]
            
        def get_weight(self) :
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight 
        
        def __call__(self,x) :
            bacth,seq = x.shape
            x = self.Embedding(x)
            if self.signal_type == 'learn' : 
                if isinstance(self.Positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.Positional_encoding(pos)
                    pos = ll.expand_dims(x=pos,axis=0)
            else  : 
                if isinstance(self.Positional_encoding,ll.GradientReflector):
                    pos = ll.expand_dims(self.Positional_encoding,axis=0)
                    pos = pos[:,:seq,:]
            x *= ll.sqrt(self.d_model)
            x += pos 

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x 
        
    
    class __Encoder_MHA_6block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate :float = 0.1,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.drop_rate = drop_rate
            self.vocab_size = vocab_size
            self.Embedding = layers.Embedding(vocab_size,self.d_model)
            self.maxpos = maxpos
            self.signal_type = signal_type
            self.block1 = layers.BlockEncoder_MHA(num_head=4,d_model=d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockEncoder_MHA(num_head=4,d_model = self.d_model,ffn = ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)

            if self.signal_type == 'learn' :
                self.PositionalEncoding = layers.Embedding(maxpos,d_model)
                self.__node_layers = [
                    self.Embedding,self.PositionalEncoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6
                ]
            else :
                self.PositionalEncoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.PositionalEncoding = ll.convert_to_tensor(self.PositionalEncoding)
                self.__node_layers = [
                    self.Embedding,self.block1,
                    self.block2,self.block3,
                    self.block4,self.block5,self.block6
                ]
        
        def get_weight(self):
            weight = []
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight 

        def __call__(self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x *= ll.sqrt(self.d_model)
            
            if self.signal_type == 'learn' :
                pos = np.arange(seq)
                pos = self.PositionalEncoding(pos)
                pos = ll.expand_dims(pos,axis=0)
            else :
                pos = ll.expand_dims(self.PositionalEncoding,axis=0)
                pos = pos[:,:seq,:]
            
            x+= pos 

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            return x 
    
    class __Encoder_MHS_9block (__BasicTransformers) :
        def __init__(self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1 ,
                     normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.drop_rate = drop_rate
            self.vocab_size = vocab_size
            self.maxpos = maxpos
            self.signal_type = signal_type
            self.Embedding = layers.Embedding(vocab_size,d_model)
            
            self.block1 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block7 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block8 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block9 = layers.BlockEncoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)

            if self.signal_type == 'learn' :
                self.Positional_encoding = layers.Embedding(
                    input_dim=self.maxpos,
                    output_dim=self.d_model
                )
                self.__node_layers = [
                    self.Embedding,self.Positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,
                    self.block7,self.block8,self.block9
                ]
            else :
                self.Positional_encoding =  ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.Positional_encoding = ll.GradientReflector(
                    self.Positional_encoding,_op='Positional'
                )
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,self.block6,
                    self.block7,self.block8,self.block9
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight 

        def __call__(self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)
            
            if self.signal_type == 'learn' :
                if isinstance(self.Positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.Positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                if isinstance(self.Positional_encoding,ll.GradientReflector):
                    pos = self.Positional_encoding
                    pos = ll.expand_dims(pos,axis=0)
            x += pos 

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)

            return x 
    
    class __Encoder_Attn_3block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,
                      maxpos,signal_type,drop_rate : float = 0.1,normMode : Literal['prenorm','postnorm']='prenorm'): 
            super().__init__()
            self.d_model = d_model 
            self.maxpos= maxpos 
            self.signal_type = signal_type
            self.vocab_size = vocab_size 
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(
                input_dim=self.vocab_size,output_dim=self.d_model
            ) 
            self.block1 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            
            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(
                    input_dim=self.maxpos,
                    output_dim=self.d_model
                )
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(
                    tensor=self.positional_encoding,_op='postional'
                )
                self.__node_layers = [
                    self.Embedding,self.block1,
                    self.block2,self.block3
                ]
            
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight

        def __call__ (self,x) :
            batch,seq= x.shape
            x = self.Embedding(x)
            x *= ll.sqrt(self.d_model)
            
            if self.signal_type == 'learn' :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            
            x+= pos 
            
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            return x 
    
    class __Encoder_Attn_6block (__BasicTransformers) :
        def __init__(self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1,
                     normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.maxpos = maxpos
            self.vocab_size = vocab_size
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            
            self.Embedding = layers.Embedding(
                input_dim=self.vocab_size,output_dim=self.d_model
            )
            self.block1 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NorMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)

            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(
                    input_dim=self.maxpos,output_dim=self.d_model
                )
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(self.positional_encoding,_op='position')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,self.block6
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
        
        def __call__(self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)
            
            if self.signal_type == 'learn' :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding 
                pos = ll.expand_dims(pos,axis=0)
                pos = pos [:,:seq,:]
            
            x+= pos 
            
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)

            return x 
    
    class __Encoder_Attn_9block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1  ,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.vocab_size = vocab_size 
            self.maxpos = maxpos 
            self.signal_type = signal_type
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.drop_rate = drop_rate
            
            self.block1 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block7 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block8 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block9 = layers.BlockEncoder_Attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)

            if self.signal_type == 'learn' :
                self.postional_encoding = layers.Embedding(
                    input_dim=self.maxpos,output_dim=self.d_model
                )
                self.__node_layers = [
                    self.Embedding,self.postional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,
                    self.block7,self.block8,self.block9
                ]
            else :
                self.postional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model = self.d_model
                )
                self.postional_encoding = ll.GradientReflector(self.postional_encoding,_op='Positional')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,self.block7,
                    self.block8,self.block9
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight 
        
        def __call__ (self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x *= ll.sqrt(self.d_model)

            if self.signal_type == 'learn' :
                if isinstance(self.postional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.postional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.postional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            
            x += pos 

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)

            return x 
    
    class __Decoder_Multi_3block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_size,maxpos,signal_type,drop_rate : float = 0.1,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.maxpos = maxpos 
            self.vocab_size = vocab_size
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_MHA(num_head=2,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_MHA(num_head=2,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_MHA(num_head=2,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')
            if self.signal_type =='learn' :
                self.positional_encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.linear
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(self.positional_encoding,_op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,
                    self.block2,self.block3,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight
        
        def __call__ (self,x) : 
            batch,seq=x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)

            if self.signal_type == 'learn' :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq) 
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.linear(x)
            return x 
    
    class __Decoder_Multi_6block (__BasicTransformers) :
        def __init__(self,d_model,vocab_size,ffn_size,maxpos,signal_type,drop_rate: float = 0.1 ,
                     normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.vocab_size = vocab_size
            self.maxpos = maxpos 
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)   
            self.block3 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)   
            self.block4 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)   
            self.block5 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate = self.drop_rate)   
            self.block6 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_size,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')

            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,
                    self.linear
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(tensor=self.positional_encoding,
                                                                _op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,self.block6
                    ,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight
        
        def __call__ (self,x) :
            batch,seq = x.shape
            x*= ll.sqrt(self.d_model)
            
            if self.signal_type =='learn' :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
            
            x+=pos 
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.linear(x)
            return x 
    
    class __Decoder_Multi_9block (__BasicTransformers) :
        def __init__(self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1 ,
                     normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.maxpos = maxpos 
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode ,drop_rate=self.drop_rate)
            self.block5 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block7 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block8 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block9 = layers.BlockDecoder_MHA(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')
            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(
                    input_dim=self.vocab_size,output_dim=self.d_model
                )
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,
                    self.block7,self.block8,self.block9
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(tensor=self.positional_encoding,_op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,
                    self.block6,self.block8,self.block9,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight

        def __call__ (self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)

            if self.signal_type == 'learn' :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0) 
                pos = pos[:,:seq,:]
            
            x += pos
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.linear(x)

            return x 
    
    class __Decoder_Attention_3block (__BasicTransformers)  :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1 ,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model
            self.maxpos = maxpos 
            self.signal_type = signal_type
            self.vocab_size = vocab_size
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')
            
            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,self.linear
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(tensor=self.positional_encoding,_op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,
                    self.block2,self.block3,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight

        def __call__ (self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)
            
            if self.signal_type == 'learn'  :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            x += pos
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.linear(x)

            return x 
    
    class __Decoder_Attention_6block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.maxpos = maxpos 
            self.vocab_size = vocab_size 
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(self.vocab_size,activation='softmax')

            if self.signal_type =='learn' :
                self.positional_Encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.positional_Encoding,
                    self.block1,self.block2,self.block3,self.block4,
                    self.block5,self.block6,self.linear
                ]
            else :
                self.positional_Encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model = self.d_model
                )
                self.positional_Encoding = ll.GradientReflector(tensor= self.positional_Encoding,
                                                                _op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,self.block6,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            
            return weight 
        
        def __call__(self,x) :
            batch,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)
            
            if self.signal_type == 'learn' :
                if isinstance(self.positional_Encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_Encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            
            else :
                pos = self.positional_Encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            x += pos 

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.linear(x)
            return x 
    
    class __Decoder_Attention_9block (__BasicTransformers) :
        def __init__(self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1 ,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model 
            self.vocab_size = vocab_size
            self.maxpos = maxpos
            self.signal_type = signal_type 
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode = normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block7 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block8 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block9 = layers.BlockDecoder_attention(d_model=self.d_model,ffn_dim=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')


            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,self.block4,
                    self.block5,self.block6,self.block7,self.block8,self.block9,
                    self.linear
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(
                    tensor=self.positional_encoding,_op='position'
                )
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,
                    self.block6,self.block7,self.block8,self.block9,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight 
        
        def __call__ (self,x) :
            b,s = x.shape 
            x = self.Embedding(x)
            x *= ll.sqrt(self.d_model) 
            if self.signal_type == 'learn' :
                if isinstance(self.positional_encoding,layers.Embedding) :
                    pos = np.arange(s)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos [:,:s,:]
            
            x+= pos 
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.linear(x)
            return x 
    
    class __DecodersCross_Multi_3block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1 ,
                      normMode : Literal['prenorm','postnorm']='prenorm' ) :
            super().__init__()
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.maxpos = maxpos
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_MHA_cross(num_head=2,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_MHA_cross(num_head=2,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_MHA_cross(num_head=2,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate = self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')
            
            if self.signal_type == 'learn' :
                self.postional_encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.postional_encoding,
                    self.block1,self.block2,self.block3,self.linear
                ]
            else :
                self.postional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.postional_encoding = ll.GradientReflector(tensor=self.d_model,_op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,
                    self.block2,self.block3,self.linear
                ]
        
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight
        
        def __call__ (self,x,cross_logits=None) :
            if cross_logits is None :
                raise RuntimeError("Cross logits can't be none, cause cross attention mechanism")
            batch,seq = x.shape
            x = self.Embedding(x)
            x *= ll.sqrt(self.d_model)

            if self.signal_type == 'learn' :
                if isinstance(self.postional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.postional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.postional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            
            x+= pos 
            x = self.block1(x,cross_logits)
            x = self.block2(x,cross_logits)
            x = self.block3(x,cross_logits)
            x = self.linear(x)
            return x 
    
    class __DecodersCross_Multi_6block(__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            self.d_model = d_model
            self.maxpos = maxpos
            self.signal_type = signal_type
            self.vocab_size = vocab_size 
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode = normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear1 = layers.Dense(units=self.vocab_size,activation='softmax')

            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(self.maxpos,self.vocab_size)
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,self.linear1
                ]
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(self.positional_encoding,
                                                                _op='position')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,
                    self.block6,self.linear1
                ]
            
        def get_weight(self):
            weight = list()
            for layers in self.__node_layers :
                wg = layers.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight
        
        def __call__ (self,x,cross_attn=None) :
            if cross_attn is None :
                raise RuntimeError("The cross_attn can't be None")

            b,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)
            if self.signal_type == 'learn' :
                if isinstance (self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            x+= pos 
            x = self.block1(x,cross_attn)
            x = self.block2(x,cross_attn)
            x = self.block3(x,cross_attn)
            x = self.block4(x,cross_attn)
            x = self.block5(x,cross_attn)
            x = self.block6(x,cross_attn)
            x = self.linear1(x)
            return x 
    class __DecodersCross_Multi_9block (__BasicTransformers) :
        def __init__ (self,d_model,vocab_size,ffn_dim,maxpos,signal_type,drop_rate : float = 0.1,
                      normMode : Literal['prenorm','postnorm']='prenorm') :
            super().__init__()
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.maxpos = maxpos 
            self.signal_type = signal_type
            self.drop_rate = drop_rate
            self.Embedding = layers.Embedding(input_dim=self.vocab_size,output_dim=self.d_model)
            self.block1 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block2 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block3 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block4 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block5 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block6 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block7 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block8 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.block9 = layers.BlockDecoder_MHA_cross(num_head=4,d_model=self.d_model,ffn=ffn_dim,NormMode=normMode,drop_rate=self.drop_rate)
            self.linear = layers.Dense(units=self.vocab_size,activation='softmax')

            if self.signal_type == 'learn' :
                self.positional_encoding = layers.Embedding(input_dim=self.maxpos,output_dim=self.d_model)
                self.__node_layers = [
                    self.Embedding,self.positional_encoding,
                    self.block1,self.block2,self.block3,
                    self.block4,self.block5,self.block6,
                    self.block7,self.block8,self.block9] 
            else :
                self.positional_encoding = ll.preprocessing.PositionalEncodingSinusoidal(
                    maxpos=self.maxpos,d_model=self.d_model
                )
                self.positional_encoding = ll.GradientReflector(tensor=self.positional_encoding,
                                                                _op='Position')
                self.__node_layers = [
                    self.Embedding,self.block1,self.block2,
                    self.block3,self.block4,self.block5,
                    self.block6,self.block7,self.block8,self.block9,self.linear
                ]
        def get_weight(self):
            weight = list()
            for node in self.__node_layers :
                wg = node.get_weight()
                if wg is not None :
                    for w in wg :
                        weight.append(w)
            return weight
        
        def __call__ (self,x,cross_attn) :
            if cross_attn is None :
                raise RuntimeError("The cross_attn can't be None")
            batch,seq = x.shape
            x = self.Embedding(x)
            x*= ll.sqrt(self.d_model)

            if self.signal_type == 'learn' :
                if isinstance(self.positional_encoding,layers.Embedding):
                    pos = np.arange(seq)
                    pos = self.positional_encoding(pos)
                    pos = ll.expand_dims(pos,axis=0)
            else :
                pos = self.positional_encoding
                pos = ll.expand_dims(pos,axis=0)
                pos = pos[:,:seq,:]
            
            x+= pos 
            x = self.block1(x,cross_attn)
            x = self.block2(x,cross_attn)
            x = self.block3(x,cross_attn)
            x = self.block4(x,cross_attn)
            x = self.block5(x,cross_attn)
            x = self.block6(x,cross_attn)
            x = self.block7(x,cross_attn)
            x = self.block8(x,cross_attn)
            x = self.block9(x,cross_attn)
            x = self.linear(x)
            return x


    def __build_model (self) : 
        if self.type not in  ['encoder-nlp','decoder-nlp','decoder-cross'] :
            raise RuntimeError("type not valid")
            
        if self.level not in ['light','balance','deep'] :
            raise RuntimeError("level not valid")
        
        if self.maxpos == 0 or self.maxpos is None  :
            raise RuntimeError("Maxpos parameter can't be None or 0")
        
        if self.Head_type not in  ['Multi','Single']  :
            raise RuntimeError("Head Type not Valid")
        
        if self.posencoding_type not in ['learn','constant'] :
            raise RuntimeError("Positional Encoding type not Valid")
        
        if self.type == 'encoder-nlp' :
            if self.Head_type == 'Multi' :
                if self.level == 'light' :

                    self.Model = self.__Encoder_MHA_3block(
                        d_model=self.d_model,
                        vocab_size=self.vocab_size,
                        ffn = self.ffn_size,
                        max_pos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )

                elif self.level == 'balance' :

                    self.Model = self.__Encoder_MHA_6block(
                        d_model=self.d_model,
                        vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )
                
                else :
                    
                    self.Model = self.__Encoder_MHS_9block(
                        d_model = self.d_model,vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,
                        drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )
            else :

                if self.level == 'light' :
                    self.Model = self.__Encoder_Attn_3block(
                        d_model=self.d_model,vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )
                
                elif self.level == 'balance' :
                    self.Model = self.__Encoder_Attn_6block(
                        d_model=self.d_model,vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )
                
                else :
                    self.Model = self.__Encoder_Attn_9block(
                        d_model=self.d_model,vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode = self.NormMode
                    )
        if self.type == 'decoder-nlp' :
            if self.Head_type == 'Multi' : 
                if self.level == 'light' :
                    self.Model = self.__Decoder_Multi_3block(
                        d_model=self.d_model,vocab_size=self.vocab_size,
                        ffn_size=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode= self.NormMode
                    )
                elif self.level == 'balance' :
                    self.Model = self.__Decoder_Multi_6block(
                        d_model=self.d_model,vocab_size=self.vocab_size,ffn_size=self.ffn_size,
                        maxpos=self.maxpos,signal_type=self.posencoding_type,drop_rate=self.drop_rate,
                        normMode=self.NormMode
                    )
                
                else :
                    self.Model = self.__Decoder_Multi_9block(
                        d_model=self.d_model,vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )
            else :
                if self.level == 'light' :
                    self.Model = self.__Decoder_Attention_3block(
                        d_model=self.d_model,vocab_size=self.vocab_size,ffn_dim=self.ffn_size,
                        maxpos=self.maxpos,signal_type=self.posencoding_type,drop_rate=self.drop_rate,
                        normMode=self.NormMode
                    )
                
                elif self.level == 'balance' :
                    self.Model = self.__Decoder_Attention_6block(
                        d_model=self.d_model,vocab_size=self.vocab_size,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,drop_rate=self.drop_rate,
                        signal_type=self.posencoding_type,
                        normMode=self.NormMode
                    )
                
                else :
                    self.Model = self.__Decoder_Attention_9block(
                        d_model=self.d_model,vocab_size=self.vocab_size,drop_rate=self.drop_rate,
                        ffn_dim=self.ffn_size,maxpos=self.maxpos,signal_type=self.posencoding_type,
                        normMode = self.NormMode 
                    )
            if self.type == 'decoder-cross' :
                if self.Head_type == 'Multi' :
                    if self.level == 'light' :
                        self.Model = self.__DecodersCross_Multi_3block(
                            d_model=self.d_model,vocab_size=self.vocab_size,drop_rate=self.drop_rate,
                            ffn_dim=self.ffn_size,maxpos=self.maxpos,signal_type=self.posencoding_type,
                            normMode=self.NormMode
                        )
                    elif self.level == 'balance' :
                        self.Model = self.__DecodersCross_Multi_6block(
                            d_model=self.d_model,vocab_size=self.vocab_size,drop_rate=self.drop_rate,
                            ffn_dim=self.ffn_size,maxpos=self.maxpos,signal_type=self.posencoding_type,
                            normMode=self.NormMode
                        )
                    else :
                        self.Model = self.__DecodersCross_Multi_9block(
                            d_model=self.d_model,vocab_size=self.vocab_size,drop_rate=self.drop_rate,
                            ffn_dim=self.ffn_size,maxpos=self.maxpos,signal_type=self.posencoding_type,
                            normMode=self.NormMode
                        )
                else :
                    raise RuntimeError("Sorry Decoder Cross for Single head type will able for next update")


    
    def __call__ (self,x) :
        """ Call model for forwardpass """
        return self.Model(x)
    
    def predict(self,x) :
        """ Predict from model"""
        return self.__call__(x)
    
    def get_weight(self) :
        """ Get weight from model"""
        return self.Model.get_weight()
            
        


        
