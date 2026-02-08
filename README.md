# 🌱 LittleLearn – Touch the Big World with Little Steps

update Version (1.1.2) date : (4-January-2026): 

    - add new layers Logits Mixing Attention
    - DivBackward bug fixed 
    - add new optimizers AdaBelief 


warning : on this update we remove so many feature because paradims changed.      



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

## 📦 Ecosystem Features
- ✅ Deep learning modules: Dense, LSTM, attention mechanisms, and more


- 🤖 instant model by Model Module

- 🔄 Custom training loops with full backend access

- 🧠 All powered by the GradientReflector engine — providing automatic differentiation with    transparency and tweakability

## 🔧 Installation

```bash
    pip install littlelearn
```

🚀 Quick Example : 
```bash
    import littlelearn as ll 
    import littlelearn.DeepLearning as dl 

    model = dl.layers.Sequential([
        dl.layers.Linear(20,32),
        dl.activations.Relu(),
        dl.layers.Linear(32,1)
    ]) 
    model.train()
    x_train,y_train= datasets()
    optimizer = dl.optimizers.Adam(model.parameter())

    for epoch in range(100) :
        y_pred = model(x_train)
        loss = dl.loss.mse_loss(y_train,y_pred)
        loss.backwardpass()
        optimizer.step()
        loss.reset_grad()
        print(loss.tensor)
    
    model.inference()
    model.save("model.npz")

``` 
📌 Disclaimer
While inspired by well-known frameworks, LittleLearn is built entirely from scratch with its own mechanics.
It is suitable for:

- 🔬 Experimental research

- 🏗️ Framework building

- 📚 Educational purposes

- 🔧 Custom low-level operations


suport this project : https://ko-fi.com/alpin92578

👤 Author
Candra Alpin Gunawan
📧 hinamatsuriairin@gmail.com
🌐 GitHub https://github.com/Airinchan818/LittleLearn

youtube : https://youtube.com/@hinamatsuriairin4596?si=KrBtOhXoVYnbBlpY