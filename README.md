# Hierarchical News Classification with CNNs and DistilBERT

An experimental natural language processing project for classifying English-language news articles at two levels of a hierarchy:

- **Level 1:** 17 broad categories
- **Level 2:** 109 fine-grained subcategories

The project compares a custom one-dimensional convolutional neural network (CNN) with fine-tuned [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) models. It was developed as a group project for the University of Thessaly course **ECE418 - Neuro-Fuzzy Computing (2023-24)**.

## Project overview

The notebook implements the complete experimental workflow:

1. Load and inspect the hierarchical news dataset.
2. Split articles into training and held-out sets.
3. Normalize and tokenize text for the CNN pipeline.
4. Train separate CNN classifiers for Levels 1 and 2.
5. Fine-tune separate DistilBERT classifiers for Levels 1 and 2.
6. Evaluate predictions with accuracy and per-class precision, recall, and F-score.
7. Visualize training curves and the Level 1 confusion matrix.

The CNN uses a trainable 100-dimensional embedding, a 640-filter `Conv1D` layer, batch normalization, Leaky ReLU activations, global max pooling, dropout, and a softmax output layer. The transformer experiments fine-tune DistilBERT for 15 epochs with AdamW.

## Recorded results

The following results are preserved in the committed notebook outputs. They have **not** been rerun from a fresh environment because the dataset is not included in this repository.

| Model | Prediction level | Labels | Evaluation split | Accuracy |
| --- | --- | ---: | --- | ---: |
| 1D CNN | Level 1 categories | 17 | Held-out test set | **75.64%** |
| 1D CNN | Level 2 subcategories | 109 | Held-out test set | **58.20%** |
| DistilBERT | Level 1 categories | 17 | Held-out validation set | **78.94%** |
| DistilBERT | Level 2 subcategories | 109 | Held-out validation set | **62.87%** |

The saved notebook records 8,733 training examples and 2,184 held-out examples, for 10,917 articles in total. It also records average single-sample CNN inference times of 35.54 ms for Level 1 and 36.51 ms for Level 2 on the original, undocumented machine. These timing values should not be compared across hardware without a controlled benchmark.

## Repository contents

```text
.
|-- NeuroFuzzy_Project.ipynb        # End-to-end exploration, training, and evaluation
|-- NeuroFuzzy_Project_Report.pdf   # Course report with methodology and discussion
|-- requirements.txt                # Direct notebook dependencies (unversioned)
|-- .gitignore                      # Local data, environments, and model artifacts
|-- LICENSE                         # GNU GPL v3
`-- README.md
```

## Dataset contract

The notebook expects a file named `news-classification.csv` in the repository root with these columns:

| Column | Meaning |
| --- | --- |
| `content` | Full article text |
| `category_level_1` | Broad category label |
| `category_level_2` | Fine-grained subcategory label |

The dataset file, its original source, and its redistribution license are not currently documented in the repository. A fresh clone therefore cannot reproduce training until the original CSV is obtained and placed at the path above. The file is ignored by Git to prevent accidental redistribution before its license is confirmed.

## Setup

The notebook metadata records Python 3.9.18. Package versions were not captured, so `requirements.txt` documents direct dependencies rather than a fully reproducible lockfile.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Place `news-classification.csv` in the repository root, then start the notebook:

```bash
jupyter lab NeuroFuzzy_Project.ipynb
```

Run the cells from top to bottom. The CNN experiments can run on a CPU, while the DistilBERT experiments are computationally expensive and are best run with a CUDA-capable GPU. The first transformer run also downloads model files from Hugging Face.

## Experimental design notes

- The CNN path uses an 80/20 train/test split with `random_state=42`, followed by a 90/10 train/validation split within the training data.
- The tokenizer is fitted only on training articles.
- The transformer path shuffles the dataframe without a fixed seed before applying its 80/20 split. Its recorded results are therefore not guaranteed to reproduce exactly.
- Separate models are trained for the two hierarchy levels; the Level 2 prediction is not constrained by the Level 1 prediction.
- Accuracy is reported for comparability with the original experiment. Macro-averaged metrics would better expose performance across imbalanced Level 1 categories.

## Limitations

- Dataset provenance and licensing are missing, which blocks complete reproduction from a fresh clone.
- Dependency versions, random seeds, hardware details, and trained checkpoints were not captured.
- The transformer experiment evaluates on its validation split and has no separate final test split.
- The notebook is monolithic and contains training, evaluation, plotting, and artifact-saving logic in the same document.
- There is no lightweight classical baseline, such as TF-IDF with logistic regression, against which to compare the neural models.

## Recommended next steps

1. Document the dataset source and license, then add a deterministic data-preparation script.
2. Define fixed train/validation/test splits and seed NumPy, TensorFlow, PyTorch, and dataframe shuffling.
3. Extract preprocessing, model construction, training, and evaluation into tested Python modules.
4. Add a TF-IDF baseline and report macro F1 alongside accuracy.
5. Add a small inference entry point and a downloadable example model or model card.
6. Add continuous integration after the reusable package and fast unit tests are on the default branch.

## Authors

- Thomas Katraouras
- Ioannis Kalamakis
- Alexandros Betsas

## License

This repository is licensed under the [GNU General Public License v3.0](LICENSE).
