"""
Laten Connected Model Architecture:
---------------
intruduction:
    Laten Connected Model Architecture is a deep learning model architecture that is based on the Laten Connected Block.
    Laten Connected Model look to relavant information with two gates prefektive sigmoid 
    and two gates result will combine as add values pass to layers with tanh activation for looking how many information 
    can look as relavant information.
    Laten Connected Model will look to past information by Residural connection mechanism, with LayerNorm being 
    stabilizer of training by two mode 'PerNorm' or 'Post Norm'.

here we have 4 variants of Laten Connected Model Architecture:
--------------------
    - 1. LatenConnectedModel5Block : 5 Laten Connected Block
    - 2. LatenConnectedModel8Block : 8 Laten Connected Block
    - 3. LatenConnectedModel12Block : 12 Laten Connected Block
    - 4. LatenConnectedModel16Block : 16 Laten Connected Block
    
    Laten Connected Model desained for solving RNN loss contextual information problem by long sequence and 
    hungry resource for training, and also solve the problem of Transformer model that need huge resource for training
    and need huge data for training, Laten Connected Model can be trained with small data and resurce. 
    even with big data and big model architecture Laten Connected Model can be trained with small resource and fast forward because 
    is just MLP stacking model with resicidual connection mechanism.

Author:
------------
 Candra Alpin Gunawan 

reference:
-------------
Candra Alpin Gunawan "LCM : A Latent-Connected MLP Architecture for Universal Deep Learning with Fast Convergence and low Computational Cost"
Zenodo 01 November 2025 
link:https://zenodo.org/records/17501400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2ZmJmNDg3LWI3MTYtNDVlNy05OWEzLTRiOTZkNGFhOTkzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJiZTNhZjBmMGJmN2NmN2EyNWYyMzRiZWI3MjJkMjcwZCJ9.1Tcsiz_aRDDHbR2MmUdf2MkcUPbyKsI88dRGsv1O3MpA-dxBMk7B4JiSvfwk0RKG9SBzV7WGHY3mnth_iEwhTg 
"""



from littlelearn import DeepLearning as dl
from littlelearn import preprocessing
import littlelearn as ll 
import numpy as np 
from typing import Literal

class LatenConnectedModel5Block :
    """
    Laten Connected Model with 5 Laten Connected Block
    ----------------
    is a model architecture that is based on the Laten Connected Block with 5 blocks stacking.
    this architecture is small variant of Laten Connected Model Architecture.
    
    parameters:
    ----------
    vocab_size : int 
        the size of vocabulary
    d_model : int
        the dimension of model
    maxpos : int
        the maximum position of input sequence
    NormMode : Literal['prenorm','postnorm']
        the mode of LayerNorm, 'prenorm' or 'postnorm', default is 'prenorm'
    laten_activation : Literal['sigmoid','gelu','swish']
        the activation function for laten gate, default is 'sigmoid'
    encoding_mode : Literal['sinusoidal','learned']
        the mode of positional encoding, 'sinusoidal' or 'learned', default is 'sinusoidal'
    drop_rate : float 
        rate of dropout mechanism     
    
    returns:
    -------
    outputs : Gradient Reflector Tensor 
        the output of Laten Connected Model with shape (batch_size, sequence_length, d_model) 
        gradient reflector tensor that can be used for training with backpropagation.
    
    Author:
    ------
    Candra Alpin Gunawan

    reference:
    -------------
    Candra Alpin Gunawan "LCM : A Latent-Connected MLP Architecture for Universal Deep Learning with Fast Convergence and low Computational Cost"\n
    Zenodo 01 November 2025 
    link:https://zenodo.org/records/17501400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2ZmJmNDg3LWI3MTYtNDVlNy05OWEzLTRiOTZkNGFhOTkzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJiZTNhZjBmMGJmN2NmN2EyNWYyMzRiZWI3MjJkMjcwZCJ9.1Tcsiz_aRDDHbR2MmUdf2MkcUPbyKsI88dRGsv1O3MpA-dxBMk7B4JiSvfwk0RKG9SBzV7WGHY3mnth_iEwhTg 
    """
    def __init__ (self,vocab_size : int ,d_model : int,maxpos : int,
                  NormMode : Literal['prenorm','postnorm'] = 'prenorm',drop_rate : float = 0.1,
                  laten_activation : Literal['sigmoid','gelu','swish'] = 'sigmoid',
                  encoding_mode : Literal['sinusoidal','learned'] = 'sinusoidal') :
        self.embedding = dl.layers.Embedding(vocab_size,d_model)
        self.d_model = d_model 
        self.block1 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation,
            drop_rate=drop_rate
        )
        self.block2 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation,
            drop_rate=drop_rate
        )
        self.block3 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation,drop_rate=drop_rate
        )
        self.block4 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation,
            drop_rate=drop_rate
        )
        self.block5 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        if encoding_mode == 'sinusoidal' :
            self.positional_encoding = preprocessing.PositionalEncodingSinusoidal(
                maxpos = maxpos,d_model = d_model
            )
            self.positional_encoding =ll. expand_dims(self.positional_encoding,axis=0)
            self.__node = [
                self.embedding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5
            ]
        elif encoding_mode == 'learned' :
            self.positional_encoding = dl.layers.Embedding(maxpos,d_model)
            self.__node = [
                self.embedding,
                self.positional_encoding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5
            ]
        else :
            raise ValueError("encoding_mode must be 'sinusoidal' or 'learned'")
    
    def get_weight(self) :
        weight = list() 
        for node in self.__node :
            wg = node.get_weight() 
            if wg is not None :
                for w in wg : 
                    weight.append(w)
        return weight
    
    def __call__(self,x) :
        S = x.shape[1]
        x = self.embedding(x)
        x*= ll.sqrt(self.d_model)
        if isinstance(self.positional_encoding,dl.layers.Embedding) :
            pos = np.arange(S)
            pos = self.positional_encoding(pos)
            pos = ll.expand_dims(pos,axis=0)
        else :
            pos = self.positional_encoding[:,:S,:]
        x = x + pos
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class LatenConnectedModel8Block :
    """
    Laten Connected Model with 8 Laten Connected Block.
    ----------------
    is a model architecture that is based on the Laten Connected Block with 8 blocks stacking
    this architecture is medium variant of Laten Connected Model Architecture.
    
    parameters:
    ----------
    vocab_size : int 
        the size of vocabulary
    d_model : int
        the dimension of model
    maxpos : int
        the maximum position of input sequence
    NormMode : Literal['prenorm','postnorm']
        the mode of LayerNorm, 'prenorm' or 'postnorm', default is 'prenorm'
    laten_activation : Literal['sigmoid','gelu','swish']
        the activation function for laten gate, default is 'sigmoid'
    encoding_mode : Literal['sinusoidal','learned']
        the mode of positional encoding, 'sinusoidal' or 'learned', default is 'sinusoidal'
    drop_rate : float
        rate of dropout mechanism 
    
    returns:
    -------
    outputs : Gradient Reflector Tensor
        the output of Laten Connected Model with shape (batch_size, sequence_length, d_model) 
        gradient reflector tensor that can be used for training with backpropagation.
    
    Author:
    ---------
    Candra Alpin Gunawan

    reference:
    -------------
    Candra Alpin Gunawan "LCM : A Latent-Connected MLP Architecture for Universal Deep Learning with Fast Convergence and low Computational Cost"\n
    Zenodo 01 November 2025 
    link:https://zenodo.org/records/17501400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2ZmJmNDg3LWI3MTYtNDVlNy05OWEzLTRiOTZkNGFhOTkzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJiZTNhZjBmMGJmN2NmN2EyNWYyMzRiZWI3MjJkMjcwZCJ9.1Tcsiz_aRDDHbR2MmUdf2MkcUPbyKsI88dRGsv1O3MpA-dxBMk7B4JiSvfwk0RKG9SBzV7WGHY3mnth_iEwhTg 
    """
    def __init__ (self,vocab_size : int ,d_model : int,maxpos : int,
                NormMode : Literal['prenorm','postnorm'] = 'prenorm',drop_rate : float = 0.1,
                laten_activation : Literal['sigmoid','gelu','swish'] = 'sigmoid',
                encoding_mode : Literal['sinusoidal','learned'] = 'sinusoidal') :
        self.embedding = dl.layers.Embedding(vocab_size,d_model)
        self.d_model = d_model 
        self.block1 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block2 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block3 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block4 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block5 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block6 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block7 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block8 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        if encoding_mode == 'sinusoidal' :
            self.positional_encoding = preprocessing.PositionalEncodingSinusoidal(
                maxpos = maxpos,d_model = d_model
            )
            self.positional_encoding = ll.expand_dims(self.positional_encoding,axis=0)
            self.__node = [
                self.embedding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
            ]
        elif encoding_mode == 'learned' :
            self.positional_encoding = dl.layers.Embedding(maxpos,d_model)
            self.__node = [
                self.embedding,
                self.positional_encoding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8
            ]
        else :
            raise ValueError("encoding_mode must be 'sinusoidal' or 'learned'")

    def get_weight(self) :
        weight = list() 
        for node in self.__node :
            wg = node.get_weight() 
            if wg is not None :
                for w in wg : 
                    weight.append(w)
        return weight

    def __call__(self,x) :
        S = x.shape[1]
        x = self.embedding(x)
        x*= ll.sqrt(self.d_model)
        if isinstance(self.positional_encoding,dl.layers.Embedding) :
            pos = np.arange(S)
            pos = self.positional_encoding(pos)
            pos = ll.expand_dims(pos,axis=0)
        else :
            pos = self.positional_encoding[:,:S,:]
        x = x + pos
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        return x

class LatenConnectedModel12Block :
    """
    Laten Connected Model with 12 Laten Connected Block.
    ----------------
    is a model architecture that is based on the Laten Connected Block with 12 blocks stacking
    this architecture is large variant of Laten Connected Model Architecture.

    parameters:
    ----------
    vocab_size : int
        the size of vocabulary
    d_model : int   
        the dimension of model
    maxpos : int   
        the maximum position of input sequence
    NormMode : Literal['prenorm','postnorm']
        the mode of LayerNorm, 'prenorm' or 'postnorm', default is 'prenorm'
    laten_activation : Literal['sigmoid','gelu','swish']
        the activation function for laten gate, default is 'sigmoid'
    encoding_mode : Literal['sinusoidal','learned']
        the mode of positional encoding, 'sinusoidal' or 'learned', default is 'sinusoidal'
    drop_rate : float 
        rate of dropout mechanism 

    returns:
    -----------
    outputs : Gradient Reflector Tensor
        the output of Laten Connected Model with shape (batch_size, sequence_length, d_model) 
        gradient reflector tensor that can be used for training with backpropagation.
    
    Author:
    --------
    Candra Alpin Gunawan 

    reference:
    -------------
    Candra Alpin Gunawan "LCM : A Latent-Connected MLP Architecture for Universal Deep Learning with Fast Convergence and low Computational Cost"\n
    Zenodo 01 November 2025 
    link:https://zenodo.org/records/17501400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2ZmJmNDg3LWI3MTYtNDVlNy05OWEzLTRiOTZkNGFhOTkzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJiZTNhZjBmMGJmN2NmN2EyNWYyMzRiZWI3MjJkMjcwZCJ9.1Tcsiz_aRDDHbR2MmUdf2MkcUPbyKsI88dRGsv1O3MpA-dxBMk7B4JiSvfwk0RKG9SBzV7WGHY3mnth_iEwhTg 

    """
    def __init__ (self,vocab_size : int ,d_model : int,maxpos : int,drop_rate : float = 0.1,
                NormMode : Literal['prenorm','postnorm'] = 'prenorm',
                laten_activation : Literal['sigmoid','gelu','swish'] = 'sigmoid',
                encoding_mode : Literal['sinusoidal','learned'] = 'sinusoidal') :
        self.embedding = dl.layers.Embedding(vocab_size,d_model)
        self.d_model = d_model 
        self.block1 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block2 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block3 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block4 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block5 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block6 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block7 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block8 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block9 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block10 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block11 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block12 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        if encoding_mode == 'sinusoidal' :
            self.positional_encoding = preprocessing.PositionalEncodingSinusoidal(
                maxpos = maxpos,d_model = d_model
            )
            self.positional_encoding = ll.expand_dims(self.positional_encoding,axis=0)
            self.__node = [
                self.embedding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
                self.block9,
                self.block10,
                self.block11,
                self.block12
            ]
        elif encoding_mode == 'learned' :
            self.positional_encoding = dl.layers.Embedding(maxpos,d_model)
            self.__node = [
                self.embedding,
                self.positional_encoding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
                self.block9,
                self.block10,
                self.block11,
                self.block12
            ]
        else :
            raise ValueError("encoding_mode must be 'sinusoidal' or 'learned'")

    def get_weight(self) :
        weight = list() 
        for node in self.__node :
            wg = node.get_weight() 
            if wg is not None :
                for w in wg : 
                    weight.append(w)
        return weight

    def __call__(self,x) :
        S = x.shape[1]
        x = self.embedding(x)
        x*= ll.sqrt(self.d_model)
        if isinstance(self.positional_encoding,dl.layers.Embedding) :
            pos = np.arange(S)
            pos = self.positional_encoding(pos)
            pos = ll.expand_dims(pos,axis=0)
        else :
            pos = self.positional_encoding[:,:S,:]
        x = x + pos
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        return x


class LatenConnectedModel16Block :
    """
        Layen Connected Model with 16 Laten Connected Block.
        ----------------
        is a model architecture that is based on the Laten Connected Block with 16 blocks stacking
        this architecture is extra large variant of Laten Connected Model Architecture.

        parameters:
        ----------
        vocab_size : int
            the size of vocabulary
        d_model : int
            the dimension of model
        maxpos : int
            the maximum position of input sequence
        NormMode : Literal['prenorm','postnorm']
            the mode of LayerNorm, 'prenorm' or 'postnorm', default is '
        laten_activation : Literal['sigmoid','gelu','swish']
            the activation function for laten gate, default is 'sigmoid'
        encoding_mode : Literal['sinusoidal','learned']
            the mode of positional encoding, 'sinusoidal' or 'learned', default is 'sinusoidal'
        drop_rate : float
            rate of dropout mechanism  
            
        returns:
        -----------
        outputs : Gradient Reflector Tensor
            the output of Laten Connected Model with shape (batch_size, sequence_length, d_model) 
            gradient reflector tensor that can be used for training with backpropagation.
        
        Author:
        --------
        Candra Alpin Gunawan

        reference:
        -------------
        Candra Alpin Gunawan "LCM : A Latent-Connected MLP Architecture for Universal Deep Learning with Fast Convergence and low Computational Cost"\n
        Zenodo 01 November 2025 
        link:https://zenodo.org/records/17501400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk2ZmJmNDg3LWI3MTYtNDVlNy05OWEzLTRiOTZkNGFhOTkzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJiZTNhZjBmMGJmN2NmN2EyNWYyMzRiZWI3MjJkMjcwZCJ9.1Tcsiz_aRDDHbR2MmUdf2MkcUPbyKsI88dRGsv1O3MpA-dxBMk7B4JiSvfwk0RKG9SBzV7WGHY3mnth_iEwhTg 

    """
    def __init__ (self,vocab_size : int ,d_model : int,maxpos : int,drop_rate : float = 0.1,
                NormMode : Literal['prenorm','postnorm'] = 'prenorm',
                laten_activation : Literal['sigmoid','gelu','swish'] = 'sigmoid',
                encoding_mode : Literal['sinusoidal','learned'] = 'sinusoidal') :
        self.embedding = dl.layers.Embedding(vocab_size,d_model)
        self.d_model = d_model 
        self.block1 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation,
            drop_rate=drop_rate
        )
        self.block2 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block3 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block4 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block5 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block6 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block7 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block8 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block9 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block10 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block11 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block12 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block13 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block14 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block15 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        self.block16 = dl.layers.LatenConnectedBlock(
            units=d_model,NormMode=NormMode,laten_activation=laten_activation
            ,drop_rate=drop_rate
        )
        if encoding_mode == 'sinusoidal' :
            self.positional_encoding = preprocessing.PositionalEncodingSinusoidal(
                maxpos = maxpos,d_model = d_model
            )
            self.positional_encoding = ll.expand_dims(self.positional_encoding,axis=0)
            self.__node = [
                self.embedding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
                self.block9,
                self.block10,
                self.block11,
                self.block12,
                self.block13,
                self.block14,
                self.block15,
                self.block16
            ]
        elif encoding_mode == 'learned' :
            self.positional_encoding = dl.layers.Embedding(maxpos,d_model)
            self.__node = [
                self.embedding,
                self.positional_encoding,
                self.block1,
                self.block2,
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
                self.block9,
                self.block10,
                self.block11,
                self.block12,
                self.block13,
                self.block14,
                self.block15,
                self.block16
            ]
        else :
            raise ValueError("encoding_mode must be 'sinusoidal' or 'learned'")

    def get_weight(self) :
        weight = list() 
        for node in self.__node :
            wg = node.get_weight() 
            if wg is not None :
                for w in wg : 
                    weight.append(w)
        return weight

    def __call__(self,x) :
        S = x.shape[1]
        x = self.embedding(x)
        x*= ll.sqrt(self.d_model)
        if isinstance(self.positional_encoding,dl.layers.Embedding) :
            pos = np.arange(S)
            pos = self.positional_encoding(pos)
            pos = ll.expand_dims(pos,axis=0)
        else :
            pos = self.positional_encoding[:,:S,:]
        x = x + pos
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        return x


