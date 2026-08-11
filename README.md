# Build and Visualize a Transformer Encoder from Scratch with PyTorch and Streamlit

## 1. Project Overview
This project focuses on building a Transformer Encoder from scratch using PyTorch (without high-level abstractions like `torch.nn.MultiheadAttention`) to explore the internal mechanics of self-attention, multi-head attention, and positional encodings. An interactive interpretability dashboard built with Streamlit will allow visual inspection of attention weights and head entropy metrics.

## 2. Project Status
Current Status: **Phase 1 — Repository initialization**

Project foundation and file structure have been initialized. Implementation of mathematical modules, model architecture, training pipeline, and Streamlit dashboard will be added incrementally through later phases.

## 3. Planned Technology Stack
- **Core Language & ML**: Python 3.10+, PyTorch, NumPy
- **Dataset & Tokenization**: Hugging Face `datasets` (Stanford IMDB), Hugging Face `transformers` (BERT Tokenizer only)
- **Data Analysis & Visualization**: Pandas, Matplotlib, Seaborn
- **Interactive UI**: Streamlit
- **Configuration & Environment**: python-dotenv, Pytest
- **Containerization**: Docker, Docker Compose

## 4. Planned Architecture
- **Scaled Dot-Product Attention**: $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ with optional padding mask support.
- **Multi-Head Attention**: Custom `MultiHeadAttention` module projecting into $Q, K, V$, splitting heads, computing parallel attention, and projecting outputs.
- **Positional Encodings**: Both Sinusoidal (deterministic) and Learned embedding approaches.
- **Transformer Encoder**: Stacked `EncoderLayer` instances featuring Pre-LN LayerNorm, Residual connections, and Feed-Forward networks.
- **Interpretability UI**: Interactive token-by-token heatmaps and Shannon entropy visualizations per layer/head.

## 5. Repository Structure
```
transformer-interpretability/
│
├── app.py                     # Streamlit interpretability app (Phase 12)
├── model.py                   # Custom Transformer Encoder (Phases 4-7)
├── train.py                   # Training pipeline (Phase 10)
├── verify.py                  # Tensor shape verification (Phase 8)
├── requirements.txt           # Dependency specifications (Phase 2)
├── Dockerfile                 # Container image specification (Phase 15)
├── docker-compose.yml         # Container compose file (Phase 15)
├── .dockerignore              # Docker exclusion rules
├── .gitignore                 # Git exclusion rules
├── .env.example               # Environment variables template
├── README.md                  # Project documentation
│
├── config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py
│
├── data/                      # Dataset documentation & cache directory
│   └── README.md
│
├── models/                    # Serialized PyTorch models
│   └── .gitkeep
│
├── snapshots/                 # Attention weight tensor snapshots
│   └── .gitkeep
│
├── logs/                      # Shannon entropy metrics logs
│   └── .gitkeep
│
├── verification/              # Shape verification JSON artifacts
│   └── .gitkeep
│
├── reports/                   # Attention head analysis reports
│   └── .gitkeep
│
├── tests/                     # Automated unit test suite
│   └── __init__.py
│
└── scripts/                   # Utility & training scripts
    └── .gitkeep
```

## 6. Dataset
- **Dataset**: Stanford IMDB Movie Review Dataset (`stanfordnlp/imdb`).
- **Loading Strategy**: Programmatically fetched via Hugging Face `datasets`. The raw dataset files are not committed to Git.

## 7. Development Workflow
The development follows a structured 17-phase implementation roadmap. Each component is developed, tested, and verified modularly before integration.

## 8. Verification Strategy
The codebase will incorporate automated verification scripts (`verify.py`), pytest unit tests (`tests/`), attention weight snapshots (`snapshots/`), entropy metrics (`logs/training_metrics.csv`), and DOM integration tests (`data-testid`) for Streamlit elements.
