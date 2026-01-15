# Transformer Encoder–Decoder with Temporal and Cross Attention

This repository implements a custom **Transformer encoder–decoder architecture** designed for sequence-to-sequence modeling with **temporal attention**, **self-attention**, and **cross-attention**.  
The core building unit is a modular `Block` that integrates text encoding, temporal aggregation across sequences, and conditional decoding.

The model is implemented in **PyTorch** and is suitable for tasks involving:
- Text-to-sequence generation
- Temporal or essay-level aggregation
- Conditional generation with hierarchical attention

---

## Architecture Overview

At a high level, the model consists of:

1. **Text Encoder**
   - Encodes token-level text embeddings with padding-aware attention.
2. **Temporal Attention Head**
   - Aggregates encoder outputs across time or document-level sequences.
3. **Decoder Stack**
   - Causal self-attention over decoder inputs
   - Cross-attention from decoder to encoded text representation
   - Position-wise feed-forward network
4. **Residual Connections + Layer Normalization**
   - Applied after every attention and feed-forward sub-layer

---

## Model Block (`Block`)

Each `Block` combines an encoder and a decoder with multiple attention mechanisms:

### Components
- `EncoderRLLM`: Transformer-style text encoder
- `MultiHA`:
  - Temporal attention (causal)
  - Decoder self-attention (causal)
  - Encoder–decoder cross-attention
- `FeedForward`: Position-wise MLP
- `LayerNorm`: Pre-normalized residual structure

---

## Forward Pass Logic

1. Encode token embeddings (`t_emb`) using padding mask
2. Mean-pool encoder outputs to obtain a sequence-level representation
3. Apply **temporal attention** across sequences/documents
4. Apply **causal self-attention** on decoder input `y`
5. Apply **cross-attention** from decoder to encoded text
6. Apply feed-forward transformation
7. Return updated embeddings and masks for downstream blocks

---

## Model Architecture Diagram

The following diagram represents the internal data flow of a single `Block`.

```mermaid
flowchart TD
    A[t_emb<br/>Token Embeddings] --> B[EncoderRLLM]
    B --> C[Encoder Output<br/>(B × T × d)]

    C --> D[Mean Pooling<br/>over T]
    D --> E[Temporal Multi-Head Attention<br/>(causal)]
    E --> F[LayerNorm + Residual]

    Y[y<br/>Decoder Input] --> G[Self Multi-Head Attention<br/>(causal)]
    G --> H[LayerNorm + Residual]

    F --> I[Cross Multi-Head Attention]
    H --> I
    I --> J[LayerNorm + Residual]

    J --> K[Feed Forward Network]
    K --> L[LayerNorm + Residual]

    L --> M[y<br/>Updated Decoder State]
    C --> N[t_emb<br/>Updated Encoder State]

