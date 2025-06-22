
# 🧠 NLP Model Timeline (1986–2025)

```html
<style>
  .timeline {
    display: grid;
    grid-template-columns: 70px auto;
    row-gap: 4px;
    font-family: sans-serif;
    font-size: 14px;
  }
  .year {
    font-weight: bold;
    white-space: nowrap;
  }
</style>
```

<div class="timeline">

**1986** — RNN – Basic sequential modeling  
**1997** — LSTM – Long-term memory via gating  
**2014** — GRU, Bi-RNN, Stacked RNN, Seq2Seq, Additive Attention (Bahdanau)  
**2015** — Multiplicative Attention (Luong)  
**2017** — Transformer – Self & Cross Attention  
**2018** — BERT, GPT‑1  
**2019** — RoBERTa, DistilBERT, ALBERT, XLNet, ERNIE, GPT‑2, MarianMT, BART  
**2020** — T5/mT5, ELECTRA, DeBERTa, PEGASUS, ViT, Reformer, Linformer, Performer, BigBird, Longformer  
**2021** — Mistral 7B, Switch Transformer, CLIP  
**2022** — FlashAttention, BLIP, Flamingo, Perceiver IO  
**2023** — GPT‑4, LLaMA‑1/2, DeepSeek (LLM + Coder)  
**2024** — LLaMA‑3.0 (Apr), LLaMA‑3.1 / Mistral L2 (Jul), DeepSeek‑V3 (Dec), R1‑Lite (Nov)  
**2025** — Jan: DeepSeek‑R1, Mar: V3‑0324, May: Devstral (Small‑2505), Mixtral (MoE)

</div>

---

## 🔹 1. RNN Family – Early Sequence Models  
**Goal**: Handle sequential data by using memory of past tokens.

- **RNN**: Recurrent Neural Network; learns temporal patterns, but suffers from vanishing gradients.  
- **LSTM**: Long Short-Term Memory; introduces gates (input, forget, output) to retain long-term dependencies.  
- **GRU**: Gated Recurrent Unit; simpler than LSTM, uses update/reset gates for efficiency.  
- **Bidirectional RNN**: Processes data in both forward and backward directions to capture full context.  
- **Stacked RNN**: Multiple RNN layers stacked to learn deeper temporal features.  
- **Encoder-Decoder**: Sequence-to-sequence model that maps input to output sequences, used in translation.

---

## 🔹 2. Attention Mechanisms – Beyond RNNs  
**Goal**: Focus on relevant parts of the input sequence during prediction.

- **Additive Attention (Bahdanau)**: Uses learnable weights via feedforward layers for alignment scoring.  
- **Multiplicative Attention (Luong)**: Uses dot-product between query and key vectors; more efficient.  
- **Cross-Attention**: Decoder queries encoder outputs, key for sequence-to-sequence models.  
- **Self-Attention**: Each token attends to all others in the sequence; core to Transformers.  
- **Flash Attention**: Highly optimized self-attention with reduced memory and faster runtime.

---

## 🔹 3. Transformer Era – Scalable Parallel Processing  
**Goal**: Use self-attention and parallelism to scale better than RNNs.

### 🔸 Standard Transformer (2017)
- **Architecture**: Encoder-Decoder with self-attention; introduced positional encoding and multi-head attention.

### 🔸 Efficient Transformers
- **BigBird / Longformer**: Handle long sequences using sparse or windowed attention patterns.  
- **Reformer**: Improves memory by replacing attention with hashing and reversible layers.  
- **Switch Transformer**: Uses sparse Mixture-of-Experts (MoE) for efficient routing.  
- **Performer / Linformer**: Reduce complexity from quadratic to linear in sequence length.  
- **Flash Attention**: Memory-efficient GPU-optimized attention mechanism.

### 🔸 Vision & Multimodal Transformers
- **ViT**: Vision Transformer; applies transformer architecture to image patches.  
- **Perceiver IO**: Can handle diverse modalities (text, image, audio) with a unified model.  
- **CLIP / BLIP / Flamingo**: Combine vision and text for tasks like image captioning and retrieval.

---

## 🔹 4. Pretrained Transformers – Generalizable Models  
**Goal**: Use self-supervised learning at scale to generalize to many tasks.

### 🔸 Encoder-only (Bidirectional)
- **BERT**: Bidirectional masked language model for contextual word embeddings.  
- **RoBERTa**: Robustly optimized BERT with more data and training steps.  
- **DistilBERT**: Smaller, faster BERT with 95% of performance.  
- **ALBERT**: Lightweight BERT with cross-layer parameter sharing.  
- **DeBERTa**: Uses disentangled attention for better token representation.  
- **ELECTRA**: Trains discriminator to detect replaced tokens instead of masking.

### 🔸 Encoder-Decoder
- **T5**: Text-to-Text Transfer Transformer; unifies all NLP tasks as text-to-text.  
- **mT5**: Multilingual T5 supporting many languages.  
- **PEGASUS**: Pretraining optimized for abstractive summarization.  
- **BART**: Denoising autoencoder + seq2seq; good for text generation.  
- **MarianMT**: Efficient multilingual translation model.

### 🔸 Hybrid Models
- **XLNet**: Combines autoregressive and autoencoding; learns all factor permutations.  
- **ERNIE**: Injects structured knowledge from knowledge graphs into BERT-style models.

---

## 🔹 5. Generative Models – Foundation of LLMs  
**Goal**: Autoregressively generate high-quality and coherent text.

- **GPT-1**: Introduced decoder-only transformer for language modeling.  
- **GPT-2**: Large-scale generative model with coherent paragraph generation.  
- **GPT-3**: Few-shot learning with 175B parameters; enabled prompt-based learning.  
- **GPT-4**: Multimodal and highly aligned LLM with better reasoning.  
- **Open-Source**: LLaMA, Mistral, DeepSeek, Falcon — efficient LLM alternatives for research and deployment.

---
