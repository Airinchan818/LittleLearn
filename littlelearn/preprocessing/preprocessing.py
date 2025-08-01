import numpy as np 
import traceback

def PositionalEncodingSinusoidal (maxpos,d_model) :

    """
    positional Encoding for Transformers Model, Transformers Model need a spisifik position sequence information
    but we know if position learn by embedding layers will make training need more computer source for look a best 
    position. \n
    PositionalEncodingSinusoidal can create static signal position. and then cause its not layers so not need to train. 
    make train to be light , and saving computer resource. \n

    How to Use : \n
    from LittleLearn.preprocessing import PositionalEncodingSinusoidal \n
    import LittleLearn as ll

    positional = PositionalEncodingSinusoidal(100,32)

    positional = ll.convert_to_tensor(positional)

    """

    try:
        if maxpos == 0 or maxpos is None :
            raise ValueError(f"maxpos == {maxpos} ")
        elif d_model == 0 or d_model is None :
            raise ValueError(f"d_model == {d_model}")
        positional = np.arange(maxpos,dtype=np.float32) [:,np.newaxis]
        dimention = np.arange(d_model,dtype=np.float32)
        div_values = np.power(10000.0,(2 * (dimention//2) / np.sqrt(d_model)))
        angle_rads = positional / div_values
        angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
        angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])
        return angle_rads
    except Exception as e :
        e.add_note("maxpos variable must initialization first == (PositonalEncoding(maxpos=your initialization values))")
        e.add_note("d_models variable must initialization firt == (PositionalEncoding(d_model=your dimention model values))")
        traceback.print_exception(type(e),e,e.__traceback__)
        raise 

class MinMaxScaller :

    """
    do scaling to be some range min for minimal values and max for maximum values default f_range = None . \n 
    
    how to use : \n 

    from LittleLearn.preprocessing import MinMaxScaller \n 
    import numpy as np \n 

    x = np.random.rand(10,32)\n 
    scaller = MinMaxScaller(f_range = (0,1))

    scaller.fit(x)\n
    scale_x = scaller.scaling(x)\n 
    inv_scale = scaller.inverse_scaling(scale_x)
    """

    def __init__ (self,f_range=None,epsilon=1e-6) :
        if f_range is not None :
            try : 
                if len(f_range) !=2 :
                    raise ValueError("Error : f_range must give 2 values at list [min_range,max_range] or (min_range,max_range)")
                self.r_min,self.r_max = f_range
            except Exception as e :
                traceback.print_exception(type(e),e,e.__traceback__)
        self.__range = f_range
        self.epsilon = epsilon 
        self.min = None 
        self.max = None 
    
    def fit(self,x) :
        self.min = np.min(x,axis=0)
        self.max = np.max(x,axis=0)
    
    def scaling (self,x) :
        try:
            if self.min is None or self.max is None :
                raise RuntimeError("You must fit scalers First")
            scaled = (x - self.min) / (self.max - self.min + self.epsilon)
            if self.__range is None :
                return scaled
            return scaled * (self.r_max - self.r_min) + self.r_min
        except Exception as e :
            e.add_note("do MinMaxScaller().fit(x) before do scaling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def fit_scaling (self,x) :
        self.fit(x) 
        scaled = (x - self.min) / (self.max - self.min + self.epsilon)
        if self.__range is None :
            return scaled
        return scaled * (self.r_max - self.r_min) + self.r_min

    def inverse_scaling (self,x) :
        try :
            if np.max(self.min) == np.min(x) or np.max(self.max) == np.max(x)\
            or np.min(x) > np.max(self.max) :
                warning = RuntimeWarning("Warning :  The Values its to large for inverse")
                print(warning)
            if self.min is None or self.max is None :
                raise RuntimeError("Error : You must fit scaller first")
            if self.__range is None :
                return x * (self.max - self.min) + self.min
            unscale = (x - self.r_min) / (self.r_max - self.r_min + self.epsilon)
            unscale = unscale * (self.max - self.min) + self.min
            return unscale
        except Exception as e :
            e.add_note("You must do MinMaxScaller().fit() first")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class StandardScaller :
    """
    do scalling data by Standar Deviation values at data.\n 
    how to use : \n 
    import numpy as np \n 
    from LittleLearn.preprocessing import StandardScaller \n 

    a = np.random.rand(10)\n 
    scaller = StandardScaller()

    scaller.fit(a)\n 

    a_scale = scaller.scalling(a) \n 
    invert_scale_a = scaller.inverse_scalling(a_scale)
    """
    def __init__ (self,epsilon=1e-6) :
        self.epsilon = epsilon
        self.std = None 
        self.mean = None 
    
    def fit(self,x) :
        self.mean = np.mean(x,axis=0,dtype=np.float32) 
        variance = np.mean((x - self.mean)**2,axis=0,dtype=np.float32)
        self.std = np.sqrt(variance)
    
    def scaling(self,x) :
        try : 
            if self.mean is None or self.std is None :
                raise RuntimeError("you must fit scaller first")
            return (x - self.mean) / (self.std + self.epsilon)
        except Exception as e :
            e.add_note("do StandardScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    
    def inverse_scaling(self,x) :
        try :
            if self.std is None or self.mean is None :
                raise RuntimeError("you must fit scaller first")
            return x * self.std + self.mean
        except Exception as e :
            e.add_note("do StandardScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def fit_scaling (self,x) :
        self.fit(x) 
        return (x - self.mean) / (self.std + self.epsilon)

class MaxAbsoluteScaller :
    """
    do scalling by maximum absoulute values from matriks or tensor. \n 
    how to use : \n 
    from LittleLearn.preprocessing import MaxAbsoluteScaller \n 
    import numpy as np \n 

    a = np.random.rand(10,32)\n 
    scaller = MaxAbsoluteScaller() \n 

    scaller.fit(a) \n 
    a_scale = scaller.scalling(a) \n 
    inc_a_scale = scaller.inverse_scalling(a_scale)
    """
    def __init__ (self,epsilon = 1e-6) :
        self.epsilon = epsilon
        self.max_abs = None 
    
    def fit(self,x) :
        abs_values = np.abs(x) + self.epsilon
        self.max_abs = np.max(abs_values,axis=0)
    
    def scaling (self,x) :
        try :
            if self.max_abs is None :
                raise RuntimeError("you must fit scaller first")
            return x / self.max_abs
        except Exception as e :
            e.add_note("do MaxAbsoluteScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def fit_scalling (self,x) :
        self.fit(x) 
        return x / self.max_abs
    
    def inverse_scalling (self,x) :
        try :
            if self.max_abs is None :
                raise RuntimeError("you must fit scaller first")
            return x * self.max_abs
        except Exception as e :
            e.add_note("do MaxAbsoluteScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class Tokenizer:

    """
    Tokenizer is preprocessing for do word or text data be a integer data \n 
    how to use  : \n 

    from LittleLearn.preprocessing import Tokenizer \n 
    tokenizer = Tokenizer() \n 
    train_sentence = [
        "I Like Cake","cat is the super cute animal in world"
    ] \n 
    tokenizer.fit_on_texts(train_sentence) \n 

    to look index to word :  \n 
    tokenizer.index_word 
    \n 

    to look word to index : \n 
    tokenizer.word_index \n 

    to convert sentence to index int : 

    without padding : 

    sequence = tokenizer.texts_to_sequences(train_sentence,padding_len = None) \n 

    with padding : 

    tokenizer,texts_to_sequences (train_sentence,padding_len=5)
    """

    def __init__(self):
        self.__word = dict()
        self.len_vocab = None
        self.counter = 1

    def fit_on_texts(self, texts):
        if not isinstance(texts, list):
            raise RuntimeError("Data must be list of strings")
        for sentence in texts:
            for word in sentence.strip().split():
                word = word.lower()
                if word not in self.__word:
                    self.__word[word] = self.counter
                    self.counter += 1
        self.len_vocab = len(self.__word) + 1 

    @property
    def word_index(self):
        return self.__word

    @property
    def index_word(self):
        return {v: k for k, v in self.__word.items()}

    def texts_to_sequences(self, texts,padding_len=None):
        if not isinstance(texts, list):
            raise RuntimeError("Data must be list of strings")
        sequences = []
        for sentence in texts:
            seq = []
            if padding_len is None:
                for word in sentence.strip().split():
                    idx = self.__word.get(word.lower(), 0)
                    seq.append(idx)
            else :
                s = sentence.lower()
                word = s.split()
                for i in range(padding_len) :
                    if i < len(word) :
                        seq.append(self.__word.get(word[i]))
                    else :
                        seq.append(0)
            sequences.append(seq)
        return np.array(sequences)


