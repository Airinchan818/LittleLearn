import jax.numpy as np
import traceback
import pandas as pd 
from typing import Literal 
import copy
import re 
from PIL import Image
import os


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
        self.name = "MinMaxScaler"
    
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
        self.name = "StandardScaler"
    
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
        self.name = "MaxAbsoluteScaler"
    
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
    
    def fit_scaling (self,x) :
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
        self.name = "Tokenizer"

    def fit_on_texts(self, texts):
        if not isinstance(texts, list):
            raise RuntimeError("Data must be list of strings")
        for sentence in texts:
            sentence = re.sub(r"[^a-zA-Z0-9\s]","",sentence)
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
                sentence = re.sub(r"[^a-zA-Z0-9\s]","",sentence)
                for word in sentence.strip().split():
                    idx = self.__word.get(word.lower(), 0)
                    seq.append(idx)
            else :
                sentence = re.sub(r"[^a-zA-Z0-9\s]","",sentence)
                s = sentence.lower()
                word = s.split()
                for i in range(padding_len) :
                   
                    if i < len(word) :
                        if word[i] is not None :
                            seq.append(self.__word.get(word[i]))
                    else :
                        seq.append(0)
            sequences.append(seq)
        if padding_len is None :
            return sequences
        else :
            return np.array(sequences,dtype=np.int32)


def pad_sequence(sequence : list[list], max_len : int = None) :
    out = list()
    if max_len is None :
        max_len = max(len(seq) for seq in sequence)

    for seq in sequence :
        start_seq = len(seq)
        for _ in range(start_seq,max_len) :
            seq.append(0)
        out.append(seq)
    
    return np.array(out,dtype=np.int32)


class LabelEncoder :
    """
        Label Encoder
        -------------
        its for extract string object class data to numeric representation

        how to use:

            enc=LabelEncoder()

        author: Candra Alpin Gunawan
    """

    def __init__ (self) :
        self.class_idx = dict()
        self.idx_class = dict()
        self.__counter = 0
        self.name = "LabelEncoder"
    
    def fit(self,data : list) :
        """
            for fit or train LabelEncoder, call it with list paramters. 
            example:\n
                data = list() <= its your data in list  
                enc = LabelEncoder()
                enc.fit(data)
        """
        if not isinstance(data,list) :
            raise RuntimeError("Data must be list")
        for ob in data :
            if ob not in self.class_idx:
                self.class_idx[ob] = self.__counter
                self.idx_class[self.__counter] = ob
                self.__counter +=1
    
    def encod (self,x : list) :
        """
            encod for extract data to numeric representation at trained Labelencoder
            example:\n 
                enc = LabelEncoder()
                enc.encod(x)
        """
        if not isinstance(x,list) :
            raise RuntimeError("x must be list")
        result = list()
        for data in x :
            result.append(self.class_idx[data])
        return np.array(result)
    
    def decod (self,x) :
        """
            decod is for decod at numeric to class representation, parameters must be numpy array
            example:\n 
                enc = LabelEncoder()
                data = np.array(data)
                enc.decod(data)
        """
        result = list()
        for i in range(len(x)) :
            result.append(self.idx_class[x[i]])
        return result 
    
    def fit_encod (self,data:list) :
        """
            you can fit and encod a data too. example:
                enc = LabelEncoder()
                enc.fit_encod(data)
        """
        self.fit(data)
        return self.encod(data)


class OneHotEncoder :
    """
        One Hot Encoder give representation index class by 1 values in zeros verctor 
        [0,1,0,0] = 1 \n 
        example:
            enc = OneHotEncoder()
            enc.fit(data)
            enc.encod(data)
            enc.decod(result)
        
        author: Candra Alpin Gunawan 
    """
    def __init__ (self) :
        self.__encoder = LabelEncoder()
        self.name = "OneHotEncoder"
    
    def fit(self,data:list):
        """
            fit for train OneHotEncoder example:\n 
                enc = OneHotEncoder()
                enc.fit(data)
                when data is list of data object 
        """
        if not isinstance(data,list) :
            raise RuntimeError("data must be list")
        self.__encoder.fit(data)
    
    def encod (self,data) :
        """
            encod this will transform class data to be representation ix class by one hot encoding 
            data = ['x','y'] = [[1,0],[0,1]]:
            \n example:\n 
                enc = OneHotEncoder()
                enc.encod(data)
        """
        if not isinstance(data,list) :
            raise RuntimeError("data must be list")
        data = self.__encoder.encod(data)
        one_hot = list()
        for item in data :
            vector = [0] * len(self.__encoder.class_idx)
            vector[item] = 1
            one_hot.append(vector)
        return np.array(one_hot)

    def decod (self,x) :
        """
            Warning : the x parameters must be x after predicted a model, you can use 
            one hot encoding result, but its take so more computer resource
        """
        if np.max(x) <=1 :
            result = list()
            for data in x :
                count = 0
                for idx in data :
                    if idx == 1:
                        result.append(count)
                        break 
                    count +=1
            result = np.array(result)
            result = self.__encoder.decod(result)
        else :
            result = self.__encoder.decod(x)
        
        return result


def label_to_onehot (x,class_num:int) :

    """
        label to encoder will exchange a label class like [0,1,2,3] to categoricall or one hot 
        representation. 
        example:\n 
            data = np.array([1,2,3])
            one_hot = label_to_onehot(data,4)
            output = [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]]
        author: Candra Alpin Gunawan 
    """
    result = list()
    for data in x :
        vector = [0] * class_num
        vector[data] = 1 
        result.append(vector)
    return np.array(result)

class AutoPreprocessing :
    """
        AutoPreprocessing 
        --------------------
        AutoPreprocessing class for preprocessing datasets (pandas DataFrame) by one function.

        parameters:
        ---------------
            scalers_variant (Literal):
                choice scaler variant: 'minmax','std','maxabsolute',None (default: std)

            encoder_string(Literal):
                choice encoder variant: 'tokenizer','label_encoder','one_hot' (default: label_encoder)
        
        how to use:
        -----------------
            autoprocess = AutoPreprocessing()\n
            datasets = pd.read_csv(datasets)\n
            autoprocess.train(datasets)\n
            datasets = autoprocess.transform(datasets)
        
        output:
        ----------
        pandas.DataFrame()

        author:
        -----------
        Candra Alpin Gunnawan
    """
    def __init__ (self,scalers_variant :Literal['minmax','std','maxabsolute',None] = 'std',
                  encoder_string : Literal ['tokenizer','label_encoder','one_hot'] = 'label_encoder') :
        
        if encoder_string == 'tokenizer' and scalers_variant is not None :
            raise RuntimeError("give None to scaler_variant for tokenizer encoder string ")
        
        else :
            self.encoder = Tokenizer()
        
        if encoder_string == 'one_hot':
            self.encoder = OneHotEncoder()
        elif encoder_string == 'label_encoder':
            self.encoder = LabelEncoder()
        elif encoder_string not in ['tokenizer','label_encoder','one_hot'] :
            raise RuntimeError(f"{encoder_string} not aivable here")
        
        if scalers_variant not in  ['minmax','std','maxabsolute',None] :
            raise RuntimeError(f"{scalers_variant} is not aivable here")
        else :
            if scalers_variant == 'std' :
                self.scaler = StandardScaller()
            elif scalers_variant == 'minmax' :
                self.scaler = MinMaxScaller(f_range=(0,1))
            elif scalers_variant == 'maxabsolute' :
                self.scaler = MaxAbsoluteScaller()
            else :
                self.scaler = None 
        
        self.train_status = False  
    
    def train (self,datasets : pd.DataFrame) :
        """
            train:
            -------------
            call function by:
                autoprocess.train(datasets)
            use this function before transform datasets 

            paramters:
            ----------------
            datasets: pd.DataFrame 

            output:
            -----------
            log training  
        """
        if not isinstance(datasets,pd.DataFrame) :
            raise RuntimeError("datasets must converted to be pandas DataFrame")
        colums = datasets.columns
        self.c_string = list()
        c_numeric = list()
        for c in colums :
            data = datasets[c].to_list()
            if isinstance(data[0],str) :
                self.encoder.fit(data)
                self.c_string.append(c)
                print(f"{self.encoder.name} has trained")
            else :
                c_numeric.append(c)
        
        if self.scaler is not None :
            self.scaler_clone = list()

            for i in range(len(c_numeric)) :
                self.scaler_clone.append(copy.deepcopy(self.scaler)) 
            
            for i in range(len(c_numeric)) :
                self.scaler_clone[i].name = c_numeric[i]
            
            for i in range(len(c_numeric)) :
                
                self.scaler_clone[i].fit(datasets[c_numeric[i]].values)
                print(f"{self.scaler_clone[i].name} has trained ")
        self.train_status = True 
    
    def transform (self,datasets : pd.DataFrame) :
        """
            transform:
            ----------------
            transform function call by:
                autoprocess(datasets)
            use it for transform raw dataframe to be preprocessed dataframe

            paramters:
            -------------
                datasets:pd.DataFrame 
            
            output:
            ---------
            pd.DataFrame()
        """
        if not isinstance(datasets,pd.DataFrame) :
            raise RuntimeError("datasets must converted to be pandas DataFrame")
        for c in datasets.columns :
            data = datasets[c].to_list()
            if isinstance(data[0],str) :
                datasets[c] = self.encoder.encod(data)
            else :
                if self.scaler is not None :
                    for scalers in self.scaler_clone :
                        if scalers.name == c :
                            datasets[c] = scalers.scaling(datasets[c].values)
                        else:
                            continue
        
        return datasets
    
    def train_transform (self,datasets : pd.DataFrame) :
        """ 
            train_transform:
            ----------------
            call train_transform for train and transform datasets in one function. 

            parameters:
                datasets:pd.DataFrame
            
            output:
            pd.DataFrame

        """
        self.train(datasets)
        return self.transform(datasets)
    

class Dataset :
    def __init__(self) :
        pass 

    def __getitem__ (self,idx) :
        raise NotImplementedError
    
    def __len__(self) :
        raise NotImplementedError

class DataLoader:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) :
        return (len(self.dataset) + self.batch_size -1 ) // self.batch_size

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))

        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start:start+self.batch_size]
            batch = [self.dataset[i] for i in batch_idx]
            yield self._collate(batch)

    def _collate(self, batch):
        x = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return x, y

def image_to_array (path,resize :int = None ,grayscale=False) :
    image =Image.open(path)
    if grayscale :
        image = image.convert("L")
    else :
        image = image.convert("RGB")
    
    if resize is not None :
        image = image.resize(resize)

    arr = np.array(image,dtype=np.float32) / 255.0
    if not grayscale:
        arr = np.transpose(arr,(2,0,1))
    return np.array(arr)


def images_from_folder(folder, resize=None):
    images = []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")

        if resize is not None:
            if isinstance(resize, int):
                img = img.resize((resize, resize))
            else:
                img = img.resize(resize)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        images.append(arr)

    return np.stack(images)

def images_from_folders(root_dir, label_map, resize=None):
    images = []
    labels = []

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if folder_name not in label_map:
            continue  

        label = label_map[folder_name]

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(folder_path, fname)
            img = Image.open(path).convert("RGB")

            if resize is not None:
                if isinstance(resize, int):
                    img = img.resize((resize, resize))
                else:
                    img = img.resize(resize)

            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # C,H,W

            images.append(arr)
            labels.append(label)

    X = np.stack(images)
    y = np.array(labels, dtype=np.int32)

    return {
        "image" : X,
        "label" : y
    }
