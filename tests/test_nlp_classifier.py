"""Unit tests for the nlp_classifier package."""

import numpy as np
import pytest

from nlp_classifier.preprocessor import TextPreprocessor, CONTRACTIONS, _CONTRACTION_RE
from nlp_classifier.models import build_cnn_model, build_multicnn_model, build_bilstm_model
from nlp_classifier.evaluation import evaluate_model, measure_inference_time


# ---------------------------------------------------------------------------
# TextPreprocessor tests
# ---------------------------------------------------------------------------


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def setup_method(self):
        self.tp = TextPreprocessor(max_num_words=1000, max_sequence_length=50)

    def test_lowercase(self):
        result = self.tp.preprocess_text(["Hello WORLD"])
        assert result[0] == result[0].lower()

    def test_digit_removal(self):
        result = self.tp.preprocess_text(["There are 123 cats and 456 dogs"])
        assert "123" not in result[0]
        assert "456" not in result[0]

    def test_contraction_expansion(self):
        result = self.tp.preprocess_text(["I can't believe we're here"])
        assert "can't" not in result[0]
        assert "we're" not in result[0]

    def test_hyphen_replacement(self):
        result = self.tp.preprocess_text(["state-of-the-art model"])
        assert "-" not in result[0]

    def test_single_char_removal(self):
        result = self.tp.preprocess_text(["I love a good day"])
        # single chars like 'a' and 'I' should be removed
        tokens = result[0].split()
        assert all(len(t) > 1 for t in tokens if t.strip())

    def test_stopword_removal(self):
        result = self.tp.preprocess_text(["the cat is on the mat"])
        # Common stopwords should be removed
        tokens = result[0].split()
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens

    def test_fit_and_vocab(self):
        self.tp.fit_on_texts(["hello world", "foo bar baz"])
        vocab = self.tp.get_vocab_size()
        assert vocab > 1  # at least padding + some tokens

    def test_preprocess_text_data_shape(self):
        self.tp.fit_on_texts(["hello world foo bar"])
        padded = self.tp.preprocess_text_data(["hello world"])
        assert padded.shape == (1, 50)

    def test_contraction_regex_matches_all_keys(self):
        for key in CONTRACTIONS:
            assert _CONTRACTION_RE.search(key) is not None


# ---------------------------------------------------------------------------
# Model builder tests
# ---------------------------------------------------------------------------


class TestModelBuilders:
    """Verify that model builders produce compilable models with correct output shape."""

    VOCAB = 500
    SEQ_LEN = 30

    def test_cnn_model_output_shape(self):
        model = build_cnn_model(
            vocab_size=self.VOCAB, num_classes=5, max_sequence_length=self.SEQ_LEN,
            embedding_dim=16, filters=32,
        )
        dummy = np.zeros((2, self.SEQ_LEN), dtype="int32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 5)

    def test_multicnn_model_output_shape(self):
        model = build_multicnn_model(
            vocab_size=self.VOCAB, num_classes=10, max_sequence_length=self.SEQ_LEN,
            embedding_dim=16, num_filters=32,
        )
        dummy = np.zeros((2, self.SEQ_LEN), dtype="int32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 10)

    def test_bilstm_model_output_shape(self):
        model = build_bilstm_model(
            vocab_size=self.VOCAB, num_classes=3, max_sequence_length=self.SEQ_LEN,
            embedding_dim=16, lstm_units=16,
        )
        dummy = np.zeros((2, self.SEQ_LEN), dtype="int32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 3)

    def test_cnn_softmax_sums_to_one(self):
        model = build_cnn_model(
            vocab_size=self.VOCAB, num_classes=5, max_sequence_length=self.SEQ_LEN,
            embedding_dim=16, filters=32,
        )
        dummy = np.ones((1, self.SEQ_LEN), dtype="int32")
        out = model.predict(dummy, verbose=0)
        np.testing.assert_almost_equal(out.sum(), 1.0, decimal=5)


# ---------------------------------------------------------------------------
# Evaluation helpers tests
# ---------------------------------------------------------------------------


class TestEvaluationHelpers:
    """Tests for evaluation utility functions."""

    def test_measure_inference_time(self):
        model = build_cnn_model(
            vocab_size=100, num_classes=3, max_sequence_length=20,
            embedding_dim=8, filters=16,
        )
        X = np.zeros((10, 20), dtype="int32")
        avg = measure_inference_time(model, X, n_samples=3)
        assert avg > 0

    def test_evaluate_model(self):
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.utils import to_categorical

        model = build_cnn_model(
            vocab_size=100, num_classes=3, max_sequence_length=20,
            embedding_dim=8, filters=16,
        )
        enc = LabelEncoder()
        enc.fit(["cat", "dog", "bird"])

        X = np.zeros((6, 20), dtype="int32")
        y_indices = np.array([0, 1, 2, 0, 1, 2])
        y_onehot = to_categorical(y_indices, 3)

        preds, trues, acc, prec, rec, fs = evaluate_model(model, X, y_onehot, enc)
        assert len(preds) == 6
        assert 0.0 <= acc <= 1.0
