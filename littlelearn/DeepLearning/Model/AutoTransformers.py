import littlelearn.DeepLearning.layers as l 
import littlelearn as ll
import math

class TransformersEncoder (l.Component) :
    def __init__ (self,vocab_size,embed_dim,depth,drop_rate,max_pos) :
        super().__init__()
        self.embedding = l.Embedding(vocab_size=vocab_size,embedding_dim=embed_dim)
        self.block = l.Sequential([l.TransformersBlock(embed_dim,embed_dim*4,type='multi',mode='Encoder') for _ in range(depth)])
        self.learn_pos = l.Embedding(max_pos,embed_dim)   
        self.scale = math.sqrt(float(embed_dim))
        self.linear = l.Linear(embed_dim,1)
    
    def forwardpass(self,x) :
        B,S = x.shape
        x = self.embedding(x)
        x = x * self.scale
        pos = ll.arange(0,S,dtype=ll.int32)
        pos = self.learn_pos(pos) 
        pos = ll.unsquezze(pos,axis=0)
        x = x + pos 
        x = self.block(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        x = ll.DeepLearning.activations.sigmoid(x)
        return x 