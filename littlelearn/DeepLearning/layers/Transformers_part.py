from littlelearn.DeepLearning import optimizers,activations,layers
import littlelearn as ll 
from typing import Literal

class Feed_forward (ll.DeepLearning.layers.Component):
    """
        Feed forward (FFN)
        -----------------
        this is a part a transformers model when the model have attention logits. the attention 
        logits will proces by Feed forward to bring information to be indenpendent token information,
        Feed forward will upgrade non linear information understanding for to next block or next layers. 

        Parameters
        ---------------
            d_model : int 
                its like unit in classic DNN layers but d_model must be same with blocktransformers d_model units,
                to bring a information as the d_model transformers model


            ffn_dim : int 
                same with units in classif layers DNN this parameters actually work to give more non linear
                information. ffn_dim always big more than d_model parameters cause at asumtion we want bring 
                logits information from attention to look more deep non linear information.
            
            drop_rate : float
                rate from dropout mechanism 
        
        example
        -------------
            layers = Feed_forward(d_model=32,ffn_dim=128)\n
            layers(x)

        Author
        ------------
        Candra Alpin Gunawan
    """
    def __init__(self,d_model : int ,ffn_dim : int ,drop_rate :float = 0.1  ) : 
        super().__init__()
        self.ffn_dim = ffn_dim 
        self.linear_l = layers.Dense(ffn_dim,activation=activations.Gelu())
        self.out_linear = layers.Dense(d_model)
        self.dropout = layers.DropOut(rate = drop_rate)
    
    def get_weight(self) : 
        """
        call it for get weight at this layers 
        """
        weight = list() 
        for layers in [self.linear_l,self.out_linear] :
            wgl = layers.get_weight()
            for w in wgl : 
                weight.append(w)
        return weight 
    
    def __call__ (self,x) : 
        x = self.linear_l(x)
        x = self.out_linear(x)
        x = self.dropout(x)
        return x 
class BlockEncoder_MHA(ll.DeepLearning.layers.Component):
    """
    Transformer-style encoder block using Multi-Head Attention (MHA) 
    and a feed-forward network (FFN).

    This block performs:
    1. Multi-Head Self-Attention
    2. Residual Connection + Layer Normalization
    3. Feed-Forward Network
    4. Residual Connection + Layer Normalization

    Args:
        num_head (int): Number of attention heads.
        d_model (int): Dimensionality of the model (hidden size).
        ffn (int): Dimensionality of the feed-forward layer.
        drop_rate (float): dropout rate
    
    example
    ---------
        layers = BlockEncoder_MHA(num_head=4,d_model=32,ffn=64)\n
        layers(x)
    
    Author
    -----------
    Candra Alpin Gunawan
    
    """

    def __init__(self, num_head: int, d_model: int, ffn: int,drop_rate: float = 0.1,
                 NormMode  : Literal['postnorm','prenorm'] = 'postnorm'):
        """
        Initialize the encoder block layers.

        Args:
            num_head (int): Number of attention heads.
            d_model (int): Dimensionality of the model (hidden size).
            ffn (int): Dimensionality of the feed-forward layer.
            NormMode (optional): Normalization mode, either 'postnorm' or 'prenorm'.
            drop_rate (float): drop out rate" 
        """
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.ffn = ffn
        self.Normode = NormMode
        self.drop_out1 = layers.DropOut(rate=drop_rate)


        self.Multihead_Attention = layers.MultiHeadAttention(
            units=self.d_model, 
            num_heads=num_head
        )

        self.feed_forward = Feed_forward(
            d_model=self.d_model, 
            ffn_dim=self.ffn,
            drop_rate=drop_rate
        )

        self.normal1 = layers.LayerNormalization()
        self.normal2 = layers.LayerNormalization()

    def get_weight(self):
        """
        Get all trainable weights of this encoder block.

        Returns:
            list: List of all trainable weight tensors.
        """
        weight = []

        for lyr in [self.Multihead_Attention, self.normal1, self.feed_forward, self.normal2]:
            wg = lyr.get_weight()
            if wg is not None:
                weight.extend(wg)
        return weight

    def __call__(self, x):
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor after self-attention and feed-forward network.
        """
        if self.Normode == 'prenorm' : 
            attnnorm = self.normal1(x) 
            attn = self.Multihead_Attention(attnnorm,attnnorm,attnnorm)
            attn = self.drop_out1(attn)
            attn = attn + x 

            ffnNorm = self.normal2(attn)
            ffn = self.feed_forward(ffnNorm)
            ffn = ffn + attn
            return ffn
        elif self.Normode == 'postnorm' :

            attn = self.Multihead_Attention(x, x, x)
            attn = self.drop_out1(attn) 
            attn = self.normal1(attn + x)


            ffn = self.feed_forward(attn)
            ffn = self.normal2(ffn + attn)
            return ffn
        else :
            raise RuntimeError("NormMode only support 'prenorm' or 'postnorm' ")

class BlockDecoder_MHA_cross(ll.DeepLearning.layers.Component):
    """
    Transformer-style decoder block with both self-attention and cross-attention.

    This block performs:
    1. Masked Multi-Head Self-Attention (causal mask applied)
    2. Residual Connection + Layer Normalization
    3. Cross-Attention (attending to encoder outputs)
    4. Residual Connection + Layer Normalization
    5. Feed-Forward Network
    6. Residual Connection + Layer Normalization

    Args:
        num_head (int): Number of attention heads.
        d_model (int): Dimensionality of the model (hidden size).
        ffn (int): Dimensionality of the feed-forward layer.
        drop_rate (float): rate of dropout mechanism. 
    
    Example
    ----------
        layers = BlockDecoder_MHA_cross(num_head=4,d_model,64,ffn=256)\n   
        layers(x)
    
    Author
    ------------
        Candra Alpin Gunawan
    """

    def __init__(self, num_head: int, d_model: int, ffn: int,drop_rate : float = 0.1,
                 NormMode  : Literal['postnorm','prenorm'] = 'postnorm'):
        """
        Initialize the decoder block layers.

        Args:
            num_head (int): Number of attention heads.
            d_model (int): Dimensionality of the model (hidden size).
            ffn (int): Dimensionality of the feed-forward layer.
            NormMode (optional): Normalization mode, either 'postnorm' or 'prenorm'.
            drop_rate (float): drop rate for dropout mechanism
        """
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.ffn = ffn
        self.Normode = NormMode
        self.drop_out1 = layers.DropOut(rate=drop_rate)
        self.drop_out2 = layers.DropOut(rate=drop_rate)
 

        self.feed_forward = Feed_forward(
            d_model=self.d_model,
            ffn_dim=self.ffn,
            drop_rate=drop_rate
        )


        self.MultiHead_attn = layers.MultiHeadAttention(
            units=self.d_model,
            num_heads=self.num_head,
            use_causal_mask=True
        )


        self.MultiHead_attn2 = layers.MultiHeadAttention(
            units=self.d_model,
            num_heads=self.num_head
        )


        self.normal1 = layers.LayerNormalization()
        self.normal2 = layers.LayerNormalization()
        self.normal3 = layers.LayerNormalization()


        self.__node_layers = [
            self.MultiHead_attn,
            self.normal1,
            self.MultiHead_attn2,
            self.normal2,
            self.feed_forward,
            self.normal3
        ]

    def get_weight(self):
        """
        Retrieve all trainable weights of this decoder block.

        Returns:
            list: List of all trainable weight tensors.
        """
        weight = []
        for lyr in self.__node_layers:
            wg = lyr.get_weight()
            if wg is not None:
                weight.extend(wg)
        return weight

    def __call__(self, x, cross_attn):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, target_seq_len, d_model).
            cross_attn (Tensor): Encoder output for cross-attention 
                                 (shape: batch_size, source_seq_len, d_model).

        Returns:
            Tensor: Output tensor after self-attention, cross-attention, 
                    and feed-forward network.
        """
        if self.Normode == 'prenorm' :
            attnnorm1 = self.normal1(x) 
            attn1 = self.MultiHead_attn(attnnorm1,attnnorm1,attnnorm1)
            attn1 = self.drop_out1(attn)
            attn1 = attn1 + x

            attnnorm2 = self.normal2(cross_attn)
            attncross = self.MultiHead_attn2(attn1,attnnorm2,attnnorm2)
            attncross = self.drop_out2(attncross) 
            attncross = attncross + attn1

            ffn = self.normal3(attncross)
            ffn = self.feed_forward(ffn)
            ffn = ffn + attncross
            return ffn 
        elif self.Normode == 'postnorm' :
            attn = self.MultiHead_attn(x, x, x)
            attn = self.drop_out1(attn)
            attn = self.normal1(attn + x)


            cross = self.MultiHead_attn2(attn, cross_attn, cross_attn)
            cross = self.drop_out2(cross)
            cross = self.normal2(cross + attn)

            ffn = self.feed_forward(cross)
            ffn = self.normal3(ffn + cross)

            return ffn
        else : 
            raise RuntimeError("NormMode only support 'prenorm' or 'postnorm' ")
        


class BlockEncoder_Attention(ll.DeepLearning.layers.Component):
    """
    Transformer-style encoder block using single-head Attention 
    (instead of Multi-Head Attention) and a feed-forward network.

    This block performs:
    1. Single Attention
    2. Residual Connection + Layer Normalization
    3. Feed-Forward Network
    4. Residual Connection + Layer Normalization

    Args:
        d_model (int): Dimensionality of the model (hidden size).
        ffn_dim (int): Dimensionality of the feed-forward layer.
        NormMode (optional): Normalization mode, either 'postnorm' or 'prenorm'.
        drop_rate (float): rate of dropout mechanism.

    Example
    -------------
        layers = BlockEncoder_Attention(d_model=32,ffn_dim=128) \n
        layers(x)
    
    Author 
    ------------
        Candra Alpin Gunawan
    """

    def __init__(self, d_model: int, ffn_dim: int,drop_rate : float = 0.1,
                 NormMode: Literal['postnorm', 'prenorm'] = 'postnorm'):
        """
        Initialize the encoder block layers.

        Args:
            d_model (int): Dimensionality of the model (hidden size).
            ffn_dim (int): Dimensionality of the feed-forward layer.
            drop_rate (float): rate of dropout mechanism. 

        """
        super().__init__()
        self.d_model = d_model
        self.ffn = ffn_dim
        self.NormMode = NormMode
        self.drop_out1 = layers.DropOut(rate=drop_rate) 


        self.feed_forward = Feed_forward(
            d_model=self.d_model,
            ffn_dim=self.ffn,
            drop_rate=drop_rate
        )


        self.Attention = layers.Attention(units=d_model)


        self.normal1 = layers.LayerNormalization()
        self.normal2 = layers.LayerNormalization()

        self.__node_layers = [
            self.Attention,
            self.normal1,
            self.feed_forward,
            self.normal2
        ]

    def get_weight(self):
        """
        Retrieve all trainable weights of this encoder block.

        Returns:
            list: List of all trainable weight tensors.
        """
        weight = []
        for lyr in self.__node_layers:
            wg = lyr.get_weight()
            if wg is not None:
                weight.extend(wg)
        return weight

    def __call__(self, x):
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor after attention and feed-forward network.
        """
        if self.NormMode == 'prenorm' :
            attnnorm = self.normal1(x)
            attn = self.Attention(attnnorm, attnnorm, attnnorm)
            attn = self.drop_out1(attn)
            attn = attn + x

            ffnNorm = self.normal2(attn)
            ffn = self.feed_forward(ffnNorm)
            ffn = ffn + attn
            return ffn
        elif self.NormMode == 'postnorm' :

            attn = self.Attention(x, x, x)
            attn = self.normal1(attn)


            ffn = self.feed_forward(attn)
            ffn = self.normal2(ffn + attn)

            return ffn
        else :
            raise RuntimeError("NormMode only support 'prenorm' or 'postnorm' ")

class BlockDecoders_Attention_cross(ll.DeepLearning.layers.Component):
    """
    Transformer-style decoder block using single-head attention for both 
    self-attention and cross-attention, followed by a feed-forward network.

    This block performs:
    1. Masked Self-Attention (causal masking enabled)
    2. Residual Connection + Layer Normalization
    3. Cross-Attention (attending to encoder outputs)
    4. Residual Connection + Layer Normalization
    5. Feed-Forward Network
    6. Residual Connection + Layer Normalization

    Args:
        d_model (int): Dimensionality of the model (hidden size).
        ffn_dim (int): Dimensionality of the feed-forward network.
        NormMode(optional): Normalization mode, either 'postnorm' or 'prenorm'.
        drop_rate(float): rate pf dropout mechanism.   
    
    Example
    ---------
        layers = BlockDecoders_Attention_cross(d_model=32,ffn_dim-128)\n
        layers(x)
    
    Author
    ---------
        Candra Alpin Gunawan
    """

    def __init__(self, d_model: int, ffn_dim: int,drop_rate : float = 0.1,
                 NormMode : Literal['postnorm', 'prenorm'] = 'postnorm'):
        """
        Initialize the decoder block layers.

        Args:
            d_model (int): Dimensionality of the model (hidden size).
            ffn_dim (int): Dimensionality of the feed-forward layer.
        """
        super().__init__()
        self.d_model = d_model
        self.ffn = ffn_dim
        self.NormMode = NormMode
        self.drop_out1 = layers.DropOut(rate=drop_rate)
        self.drop_out2 = layers.DropOut(rate=drop_rate) 

        self.feed_forward = Feed_forward(
            d_model=d_model,
            ffn_dim=self.ffn,
            drop_rate=drop_rate
        )

        self.Attention1 = layers.Attention(
            d_model,
            Masking=True
        )


        self.Attention2 = layers.Attention(
            d_model,
            Masking=False
        )


        self.normal1 = layers.LayerNormalization()
        self.normal2 = layers.LayerNormalization()
        self.normal3 = layers.LayerNormalization()

        self.__node_layers = [
            self.Attention1,
            self.normal1,
            self.Attention2,
            self.normal2,
            self.feed_forward,
            self.normal3
        ]

    def get_weight(self):
        """
        Retrieve all trainable weights of this decoder block.

        Returns:
            list: List of all trainable weight tensors.
        """
        weight = []
        for lyr in self.__node_layers:
            wg = lyr.get_weight()
            if wg is not None:
                weight.extend(wg)
        return weight

    def __call__(self, x, cross_attn):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, target_seq_len, d_model).
            cross_attn (Tensor): Encoder output tensor for cross-attention 
                                 (shape: batch_size, source_seq_len, d_model).

        Returns:
            Tensor: Output tensor after attention layers and feed-forward network.
        """
        if self.NormMode == 'prenorm' :
            attnnorm1 = self.normal1(x)
            attn1 = self.Attention1(attnnorm1, attnnorm1, attnnorm1)
            attn1 = self.drop_out1(attn)
            attn1 = attn1 + x

            attnnorm2 = self.normal2(cross_attn)
            attncross = self.Attention2(attn1, attnnorm2, attnnorm2)
            attncross = self.drop_out2(attncross)
            attncross = attncross + attn1

            ffnNorm = self.normal3(attncross)
            ffn = self.feed_forward(ffnNorm)
            ffn = ffn + attncross
            return ffn
        elif self.NormMode == 'postnorm' :
            attn = self.Attention1(x, x, x)
            attn = self.drop_out1(attn)
            attn = self.normal1(attn + x)

            cross = self.Attention2(attn, cross_attn, cross_attn)
            cross = self.drop_out2(cross)
            cross = self.normal2(cross + attn)

            ffn = self.feed_forward(cross)
            ffn = self.normal3(ffn + cross)

            return ffn
        else :
            raise RuntimeError("NormMode only support 'prenorm' or 'postnorm' ")
        

class BlockDecoder_MHA(ll.DeepLearning.layers.Component):
    """
    Transformer-style decoder block with Multi-Head Self-Attention (MHA) 
    followed by a feed-forward network (FFN).

    This block performs:
    1. Masked Multi-Head Self-Attention (causal masking applied for autoregressive decoding)
    2. Residual Connection + Layer Normalization
    3. Feed-Forward Network
    4. Residual Connection + Layer Normalization

    Args:
        num_head (int): Number of attention heads.
        d_model (int): Dimensionality of the model (hidden size).
        ffn (int): Dimensionality of the feed-forward layer.
        NormMode (optional): Normalization mode, either 'postnorm' or 'prenorm'.
        drop_rate(float): rate of dropout mechanism.
    
    Example
    ----------
        layers = BlockDecoder_MHA(num_head=4,d_model=32,ffn=128)\n
        layers(x)

    Author
    ----------
        Candra Alpin Gunawan
    """

    def __init__(self, num_head: int, d_model: int, ffn: int,drop_rate : float = 0.1,
                 NormMode : Literal['postnorm', 'prenorm'] = 'postnorm'):
        """
        Initialize the decoder block layers.

        Args:
            num_head (int): Number of attention heads.
            d_model (int): Dimensionality of the model (hidden size).
            ffn (int): Dimensionality of the feed-forward layer.
            drop_rate(float): rate of dropout mechanism. 
        """
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.ffn = ffn
        self.NormMode = NormMode
        self.drop_out1 = layers.DropOut(rate=drop_rate) 

        self.feed_forward = Feed_forward(
            self.d_model,
            ffn_dim=self.ffn,
            drop_rate=drop_rate
        )


        self.MultiHead_attn = layers.MultiHeadAttention(
            d_model,
            num_heads=self.num_head,
            use_causal_mask=True  
        )


        self.normal1 = layers.LayerNormalization()
        self.normal2 = layers.LayerNormalization()

        self.__node_layers = [
            self.MultiHead_attn,
            self.normal1,
            self.feed_forward,
            self.normal2
        ]

    def get_weight(self):
        """
        Retrieve all trainable weights of this decoder block.

        Returns:
            list: List of all trainable weight tensors.
        """
        weight = []
        for lyr in self.__node_layers:
            wg = lyr.get_weight()
            if wg is not None:
                weight.extend(wg)
        return weight

    def __call__(self, x):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, target_seq_len, d_model).

        Returns:
            Tensor: Output tensor after masked multi-head self-attention 
                    and feed-forward network.
        """
        if self.NormMode == 'prenorm' :
            attnnorm = self.normal1(x)
            attn = self.MultiHead_attn(attnnorm, attnnorm, attnnorm)
            attn = self.drop_out1(attn)
            attn = attn + x

            ffnNorm = self.normal2(attn)
            ffn = self.feed_forward(ffnNorm)
            ffn = ffn + attn
            return ffn
        elif self.NormMode == 'postnorm' :
    
            attn = self.MultiHead_attn(x, x, x)
            attn = self.normal1(attn + x)


            ffn = self.feed_forward(attn)
            ffn = self.normal2(ffn + attn)

            return ffn
        else : 
            raise RuntimeError("NormMode only support 'prenorm' or 'postnorm' ")

class BlockDecoder_attention(ll.DeepLearning.layers.Component):
    """
    A Transformer Decoder Block with Attention and Feed-Forward Network.

    This class implements a single decoder block used in transformer-based 
    architectures. It applies self-attention, residual connections, 
    normalization, and a position-wise feed-forward network.

    Attributes:
        d_model (int): Dimensionality of the model embeddings.
        ffn (int): Dimensionality of the feed-forward network.
        feed_forward (Feed_forward): Feed-forward sublayer instance.
        Attention (layers.Attention): Multi-head attention mechanism.
        normal1 (layers.LayerNormalization): First normalization layer.
        normal2 (layers.LayerNormalization): Second normalization layer.
        drop_rate (float): rate of dropout mechanism.  
    
    Example
    ---------
        layers = BlockDecoder_attention(d_model=32,ffn_dim=128)\n 
        layers(x)
    
    Author
    ---------
    Candra Alpin Gunawan
    """

    def __init__(self, d_model : int, ffn_dim : int,drop_rate : float = 0.1,
                 NormMode : Literal['postnorm', 'prenorm'] = 'postnorm'):
        """
        Initializes the decoder block.

        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_dim (int): Dimension of the feed-forward network.,
        NormMode (optional): Normalization mode, either 'postnorm' or 'prenorm'.
        """
        super().__init__()
        self.d_model = d_model
        self.ffn = ffn_dim
        self.NormMode = NormMode
        self.drop_out1 = layers.DropOut(rate=drop_rate)
        

        self.feed_forward = Feed_forward(self.d_model, ffn_dim=self.ffn,drop_rate=drop_rate)
        

        self.Attention = layers.Attention(self.d_model, Masking=True)
        

        self.normal1 = layers.LayerNormalization()
        self.normal2 = layers.LayerNormalization()


        self.__node_layers = [
            self.Attention,
            self.normal1,
            self.feed_forward,
            self.normal2
        ]

    def get_weight(self):
        """
        Collects and returns all trainable weights from this block.

        Returns:
            list: A list of weight tensors from each sublayer.
        """
        weight = []
        for layer in self.__node_layers:
            wg = layer.get_weight()
            if wg is not None:
                for w in wg:
                    weight.append(w)
        return weight

    def __call__(self, x):
        """
        Forward pass through the decoder block.

        Args:
            x (tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            tensor: Output tensor after attention and feed-forward layers.
        """
        if self.NormMode == 'prenorm' :
            attnnorm = self.normal1(x)
            attn = self.Attention(attnnorm, attnnorm, attnnorm)
            attn = self.drop_out1(attn)
            attn = attn + x

            ffnNorm = self.normal2(attn)
            ffn = self.feed_forward(ffnNorm)          
            ffn = ffn + attn
            return ffn
        elif self.NormMode == 'postnorm' :
        
            attn = self.Attention(x, x, x)
            attn = self.drop_out1(attn)
            attn = self.normal1(attn + x)

            ffn = self.feed_forward(attn)
            ffn = self.normal2(ffn + attn)

            return ffn
        else : 
            raise RuntimeError("NormMode only support 'prenorm' or 'postnorm' ")

