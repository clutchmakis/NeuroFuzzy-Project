# NLP-Classifier

Text Classification with Neural Networks — a modular pipeline for two-level news-article classification.

## Project Structure

```
NLP-Classifier/
├── nlp_classifier/          # Reusable Python package
│   ├── __init__.py
│   ├── data_loader.py       # DataLoader class (CSV → train/test splits)
│   ├── preprocessor.py      # TextPreprocessor (cleaning, tokenisation, padding)
│   ├── models.py            # Model builders: CNN, Multi-CNN, BiLSTM
│   └── evaluation.py        # Metrics, inference timing, plotting helpers
├── tests/
│   └── test_nlp_classifier.py   # Unit tests (pytest)
├── NeuroFuzzy_Project.ipynb # Main notebook (end-to-end pipeline)
└── README.md
```

## Available Models

| Model | Description |
|---|---|
| **CNN** | Single-kernel 1-D convolution with BatchNorm + GlobalMaxPooling |
| **Multi-CNN** | Parallel Conv1D branches (kernel sizes 2, 3, 4) for multi-scale n-gram capture |
| **BiLSTM** | Bidirectional LSTM capturing long-range dependencies in both directions |
| **DistilBERT** | Transfer-learning with `distilbert-base-uncased` (Transformers section) |

## Quick Start

```bash
# Install dependencies
pip install tensorflow scikit-learn nltk pandas matplotlib transformers datasets torch pytest

# Run unit tests
python -m pytest tests/ -v

# Open the notebook
jupyter notebook NeuroFuzzy_Project.ipynb
```

## Running Tests

```bash
python -m pytest tests/ -v
```