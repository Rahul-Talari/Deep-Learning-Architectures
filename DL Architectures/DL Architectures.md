
# 🧠 Types of Neural Network Architectures

## 📊 Architecture Comparison

| Type       | Layers   | Parameters   | Train Time       | Complexity  | Applications                         |
|------------|----------|--------------|------------------|-------------|--------------------------------------|
| Shallow    | 1–3      | 1K–10K        | Minutes–Hours    | Simple      | Linear Regression, Binary Classifier, Feature Extraction |
| Medium     | 4–10     | 100K–1M       | Hours–Days       | Moderate    | Image Classification (CNN), Language Modeling (RNN), Speech Recognition |
| Deep       | 11–50    | 1M–10M        | Days–Weeks       | High        | Object Detection (YOLO), Segmentation (FCN), NLP |
| Very Deep  | 51–100+  | 10M–100M+     | Weeks–Months     | Very High   | SOTA Vision Tasks, Advanced NLP (QA, MT) |

---

## ✅ Universal Approximation Theorem
A shallow neural network (with one hidden layer) can mimic any complex function accurately, **if it has enough neurons and a suitable activation function.**

---

# 📦 Deep Learning Architectures

## 🔹 1. ANN Architectures
- **McCulloch-Pitts**: Binary threshold model.
- **Hebbian Network**: “Neurons that fire together, wire together.”
- **Perceptron / MLP**: Basic/stacked feedforward layers.
- **ADALINE / MADALINE**: Linear adaptive models.
- **Backpropagation**: Error-driven learning for MLPs.
- **RBF Networks**: Use Gaussian activations.

---

## 🔹 2. Vision Models
- **Image Classification**: Label entire image.
- **Object Detection**: Locate + classify objects.
- **Image Segmentation**: Semantic, Instance, Panoptic segmentation.

---

## 🔹 3. NLP Models
- **RNN Family**: RNN, LSTM, GRU, etc.
- **Transformer**: Self-attention architecture.
- **Pretrained Models**: BERT, RoBERTa, T5, etc.
- **LLMs**: GPT, LLaMA — autoregressive models.

---

## 🔹 4. Generative Models
- **Autoencoders (AE, VAE)**: Latent space representation and reconstruction.
- **GANs**: Adversarial training for realistic data generation.
- **Flow Models**: RealNVP, Glow — invertible transformations.
- **Diffusion Models**: DDPM, Stable Diffusion — stepwise denoising.

