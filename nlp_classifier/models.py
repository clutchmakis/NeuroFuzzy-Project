"""Neural-network model builders for text classification.

Available architectures
-----------------------
* ``build_cnn_model`` – single-kernel-size 1-D CNN (original architecture).
* ``build_multicnn_model`` – parallel multi-kernel 1-D CNN (kernel sizes 2, 3, 4).
* ``build_bilstm_model`` – Bidirectional LSTM with optional attention-like pooling.
"""

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Original single-kernel CNN (cleaned up)
# ---------------------------------------------------------------------------

def build_cnn_model(
    vocab_size: int,
    num_classes: int,
    max_sequence_length: int = 600,
    embedding_dim: int = 100,
    filters: int = 640,
    kernel_size: int = 2,
    dropout_conv: float = 0.4,
    dropout_dense: float = 0.3,
    learning_rate: float = 7e-4,
) -> keras.Model:
    """Build and compile the original CNN text-classifier.

    Returns a compiled ``keras.Sequential`` model.
    """
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
                trainable=True,
            ),
            keras.layers.Conv1D(filters, kernel_size),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dropout(dropout_conv),
            keras.layers.Dense(filters),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(dropout_dense),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.5),
        metrics=[
            "categorical_accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Multi-kernel CNN – parallel convolutions with different kernel sizes
# ---------------------------------------------------------------------------

def build_multicnn_model(
    vocab_size: int,
    num_classes: int,
    max_sequence_length: int = 600,
    embedding_dim: int = 128,
    num_filters: int = 256,
    kernel_sizes: tuple = (2, 3, 4),
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile a multi-kernel-size CNN (Functional API).

    Three parallel ``Conv1D`` branches with different kernel sizes are
    concatenated after global max-pooling, giving the model the ability to
    capture n-gram features of different lengths simultaneously.
    """
    inp = keras.layers.Input(shape=(max_sequence_length,))
    emb = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        trainable=True,
    )(inp)

    conv_blocks = []
    for ks in kernel_sizes:
        c = keras.layers.Conv1D(num_filters, ks, padding="valid")(emb)
        c = keras.layers.BatchNormalization()(c)
        c = keras.layers.LeakyReLU()(c)
        c = keras.layers.GlobalMaxPooling1D()(c)
        conv_blocks.append(c)

    concat = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    x = keras.layers.Dropout(dropout_rate)(concat)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(dropout_rate * 0.75)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            "categorical_accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Bidirectional LSTM
# ---------------------------------------------------------------------------

def build_bilstm_model(
    vocab_size: int,
    num_classes: int,
    max_sequence_length: int = 600,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    dropout_rate: float = 0.4,
    recurrent_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile a Bidirectional LSTM text-classifier.

    The model concatenates the last hidden states from both directions,
    followed by dense layers with batch normalisation and dropout.
    """
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
                trainable=True,
            ),
            keras.layers.Bidirectional(
                keras.layers.LSTM(lstm_units, return_sequences=False, dropout=recurrent_dropout)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(256),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(dropout_rate * 0.75),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            "categorical_accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    return model
