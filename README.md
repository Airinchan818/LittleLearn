# 🌱 LittleLearn – Touch the Big World with Little Steps

update Version (0.1.6) date : (25-August-2025):

    - Gradient Vanishing at LSTM bug fixed 
    - Gradient Vanishing at GRU bug fixed 
    - mismatch backwardpas Multiple bug fixed 
    - SparseCategoricallCrossentropy suport Multi output 
    - CategoricallCrossentropy suport Multi output 
    - AutoTransformers is aivable now for NLP 
    - add new funtion expand_dims 
    - update new preprocessing tools LabelEncoder,OneHotEncoder,label_to_onehot
    - update new Model AutoBuildModel => LSTM,GRU Models Sentiment Regression with Tanh
    - update new layers BlockEncoder and BlockDecoder for make Transformers model more easy
    - update new layers Feed Forward Network for make Transformers models more easy 
    - add in line Documentation at Grad Engine for Layers Backend and loss function


LittleLearn is an experimental and original machine learning framework built from scratch — inspired by the simplicity of Keras and the flexibility of PyTorch, yet designed with its own architecture, philosophy, and gradient engine.

## 🧠 What Makes LittleLearn Different?
- 🔧 Not a wrapper – LittleLearn is not built on top of TensorFlow, PyTorch, or other major ML libraries.

- 💡 Fully original layers, modules, and autodiff engine (GradientReflector).

- 🧩 Customizable down to the node level: build models from high-level APIs or go low-level for complete control.

- 🛠️ Features unique like:

- Node-level gradient clipping

- Inline graph tracing

- Custom attention mechanisms (e.g., Multi-Head Attention from scratch)


- 🤯 Designed for both research experimentation and deep learning education.

## ⚙️ Core Philosophy
Touch the Big World with Little Steps.
Whether you want rapid prototyping or total model control — LittleLearn gives you both.

LittleLearn provides multiple levels of abstraction:

| Usage Style               | Tools Available                           |
|--------------------------|-------------------------------------------|
| 💬 One-liner models      | `AutoBuildModel`, `AutoTransformers` |
| ⚙️ Modular models        | `Sequential`, `ModelByNode` (soon)        |
| 🔬 Low-level experiment  | Layers, Loss, Optimizer manual calls      |
| 🧠 Custom gradients      | `GradientReflector` engine backend        |


## 📦 Ecosystem Features
- ✅ Deep learning modules: Dense, LSTM, attention mechanisms, and more

- 🧮 Classical ML components (in progress)

- 🤖 Automated tools like AutoBuildModel

- 🔄 Custom training loops with full backend access

- 🧠 All powered by the GradientReflector engine — providing automatic differentiation with    transparency and tweakability

## 🔧 Installation

```bash
    pip install littlelearn
```

🚀 Quick Example : 
```bash
import littlelearn as ll 

x_train = 'your datasets'
y_train = 'your target'

model = ll.DeepLearning.Model.AutoBuildModel(type='mlp-binaryclassification',level='balance')
model.fit(x_train,y_train.reshape(-1,1),epochs=10,verbose=1)
```
With AutoTransformers :
```bash
    from littlelearn import DeepLearning as dl 
    optimizer = dl.optimizers.Adam()
    loss = dl.loss.SparseCategoricallCrossentropy()
    transformers_model = dl.Model.AutoTransformers(
        d_model=128,vocab_size=10000,ffn_size=512,
        maxpos=100,type='decoder-nlp',level='balance',
        Head_type='Multi',PosEncoding='learn'
    )
    x_train,y_train = "your datasets "
    
    for epoch in range(100) :
        outputs = transformers_model(x_train)
        l = loss(y_train,outputs)
        l.AutoClipGradient()
        l.backwardpass()
        optimizer.apply_weight(transformers_model.get_weight())
        optimizer.forward_in_weight()
        l.kill_grad()
        print(f"epoch {epoch + 1} || loss : {l.get_tensor()}")

```
📌 Disclaimer
While inspired by well-known frameworks, LittleLearn is built entirely from scratch with its own mechanics.
It is suitable for:

- 🔬 Experimental research

- 🏗️ Framework building

- 📚 Educational purposes

- 🔧 Custom low-level operations

This is an Beta-stage project — expect bugs, sharp edges, and lots of potential.

👤 Author
Candra Alpin Gunawan
📧 hinamatsuriairin@gmail.com
🌐 GitHub https://github.com/Airinchan818/LittleLearn

youtube : https://youtube.com/@hinamatsuriairin4596?si=KrBtOhXoVYnbBlpY