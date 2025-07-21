# 🌱 LittleLearn – Touch the Big World with Little Steps

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
| 💬 One-liner models      | `AutoBuildModel`, `AutoTransformers` (soon) |
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
import LittleLearn as ll 

x_train = 'your datasets'
y_train = 'your target'

model = ll.DeepLearning.Model.AutoBuildModel(type='mlp-binaryclassification',level='balance')
model.fit(x_train,y_train.reshape(-1,1),epochs=10,verbose=1)
```
📌 Disclaimer
While inspired by well-known frameworks, LittleLearn is built entirely from scratch with its own mechanics.
It is suitable for:

- 🔬 Experimental research

- 🏗️ Framework building

- 📚 Educational purposes

- 🔧 Custom low-level operations

This is an alpha-stage project — expect bugs, sharp edges, and lots of potential.

👤 Author
Candra Alpin Gunawan

📧 hinamatsuriairin@gmail.com

🌐 GitHub https://github.com/Airinchan818/LittleLearn
