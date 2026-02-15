"""Text preprocessing, tokenization and padding utilities."""

import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Download required NLTK data (idempotent)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------------------------------------------------------------------------
# Contraction map – defined once at module level for efficiency
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "'s": " is",
    "\u2019s": " is",
    "'re": " are",
    "\u2019re": " are",
    "'ve": " have",
    "\u2019ve": " have",
    "n't": " not",
    "n\u2019t": " not",
    "'ll": " will",
    "\u2019ll": " will",
    "'d": " would",
    "\u2019d": " would",
    "'m": " am",
    "\u2019m": " am",
    "won't": "will not",
    "won\u2019t": "will not",
    "can't": "cannot",
    "can\u2019t": "cannot",
}

# Pre-compile a single regex that matches any contraction key
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


class TextPreprocessor:
    """Preprocess, tokenize and pad text data for neural-network input.

    Parameters
    ----------
    max_num_words : int
        Maximum vocabulary size for the tokenizer.
    max_sequence_length : int
        Length to which every sequence is padded / truncated.
    """

    CUSTOM_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\u201c\u201d\u2018\u2019'

    def __init__(self, max_num_words: int = 300_000, max_sequence_length: int = 600):
        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(filters=self.CUSTOM_FILTERS, num_words=self.max_num_words)
        self.stop_words = set(stopwords.words("english"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_on_texts(self, texts):
        """Preprocess *texts* and fit the internal tokenizer."""
        texts = self.preprocess_text(texts)
        self.tokenizer.fit_on_texts(texts)

    def preprocess_text(self, texts):
        """Clean and normalise an iterable of raw text strings."""
        preprocessed = []
        for text in texts:
            text = text.lower()
            text = re.sub(r"[-\u2013\u2014]+", " ", text)  # hyphens / dashes → space
            text = re.sub(r"\d+", "", text)  # remove digits

            # Expand contractions using single compiled regex
            text = _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)

            # Remove possessive 's
            text = re.sub(r"('s|\u2019s)\b", "", text)

            # Sentence-level stopword removal
            sentences = sent_tokenize(text)
            processed = " ".join(self._remove_stopwords(s) for s in sentences)

            # Remove single-character tokens
            processed = re.sub(r"\b\w\b", "", processed)
            preprocessed.append(processed)
        return preprocessed

    def preprocess_text_data(self, texts):
        """Convert raw texts to padded integer sequences."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_sequence_length)

    def get_vocab_size(self) -> int:
        """Return vocabulary size (includes padding index 0)."""
        return len(self.tokenizer.word_index) + 1

    def plot_token_concentration(self, num_tokens: int = 20):
        """Bar-chart of the *num_tokens* most frequent tokens."""
        sorted_counts = sorted(
            self.tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True
        )
        tokens = [t for t, _ in sorted_counts[:num_tokens]]
        freqs = [f for _, f in sorted_counts[:num_tokens]]

        plt.figure(figsize=(10, 6))
        plt.bar(tokens, freqs)
        plt.xlabel("Token")
        plt.ylabel("Frequency")
        plt.title(f"Top {num_tokens} Tokens in Vocabulary")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remove_stopwords(self, text: str) -> str:
        return " ".join(w for w in text.split() if w not in self.stop_words)
