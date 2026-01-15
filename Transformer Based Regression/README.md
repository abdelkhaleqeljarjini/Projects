# Transformer Encoder–Decoder with Temporal and Cross Attention for Regression

This repository implements a custom **Transformer encoder–decoder architecture** designed for sequence-to-sequence modeling with **temporal attention**, **self-attention**, and **cross-attention**.  
The model is implemented in **PyTorch** and is suitable for temporal text-to-number generation

---

## Architecture Overview

At a high level, the model consists of:

1. **Text Encoder**
   - Encodes token-level text embeddings with padding-aware attention.
2. **Multi-head Temporal Attention Head**
   - Aggregates encoder outputs across time sequences.
3. **Decoder Stack**
   - Multi-head Causal self-attention over decoder inputs
   - Multi-head Cross-attention from decoder to encoded text representation
   - Position-wise feed-forward network
4. **Residual Connections + Layer Normalization**
   - Applied after every attention and feed-forward sub-layer
```mermaid
flowchart LR

  %% Inputs
  T[text embedding] --> E0[EncoderRLLM]
  Y[y] --> SA[Multi-head 
  Self_causal_Attention]

  %% Repeated block
  subgraph B["Encoder-Decoder Block × 2"]
    direction LR

    E0 --> P[Mean_Pooling]
    P --> TA[Multi-head
Temporal_causal_Attention]
    TA --> LN1[Add_LayerNorm]

    SA --> LN2[Add_LayerNorm]

    LN1 --> CA[Multi-head
Cross_Attention]
    LN2 --> CA
    CA --> LN3[Add_LayerNorm]

    LN3 --> FFN[Feed_Forward]
    FFN --> LN4[Add_LayerNorm]
  end

  %% Outputs
  LN4 --> O1[Linear]
  O1 --> O2[Softmax]
  O2 --> O3["probabilities over decoder vocab_size"]:::text
  classDef text fill:none,stroke:none,color:#333;
```

---

## Forward Pass Logic

1. Encode token embeddings (`text embedding`) using padding mask
2. Mean-pool encoder outputs to obtain a sequence-level representation
3. Apply **temporal attention** across sequences
4. Apply **causal self-attention** on decoder input `y`
5. Apply **cross-attention** from decoder to encoded text
6. Apply feed-forward transformation
7. Return updated embeddings and masks for downstream blocks

---




