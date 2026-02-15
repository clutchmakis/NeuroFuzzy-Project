"""Data loading and splitting utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load a CSV dataset and split into train/test sets for two-level classification.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.
    text_column : str
        Column containing the text data.
    label_column_1 : str
        Column for the primary (level-1) category.
    label_column_2 : str
        Column for the secondary (level-2) category.
    test_size : float
        Fraction of data to reserve for testing.
    """

    def __init__(
        self,
        data_path: str,
        text_column: str = "content",
        label_column_1: str = "category_level_1",
        label_column_2: str = "category_level_2",
        test_size: float = 0.2,
    ):
        self.data_path = data_path
        self.text_column = text_column
        self.label_column_1 = label_column_1
        self.label_column_2 = label_column_2
        self.test_size = test_size

    def load_data(self):
        """Read the CSV and return train/test splits with unique label arrays.

        Returns
        -------
        tuple
            X_train, y1_train, y2_train, X_test, y1_test, y2_test, y1_labels, y2_labels
        """
        df = pd.read_csv(self.data_path)

        X = df[self.text_column].values
        y1 = df[self.label_column_1].values
        y2 = df[self.label_column_2].values

        y1_labels = np.array(df[self.label_column_1].unique())
        y2_labels = np.array(df[self.label_column_2].unique())

        X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
            X, y1, y2, test_size=self.test_size, random_state=42
        )

        return X_train, y1_train, y2_train, X_test, y1_test, y2_test, y1_labels, y2_labels

    def describe(self):
        """Print descriptive statistics about the dataset."""
        df = pd.read_csv(self.data_path)

        word_counts = df[self.text_column].apply(lambda x: len(str(x).split()))
        print(f"Average words per text: {word_counts.mean():.1f}")
        print(f"Median words per text: {word_counts.median():.1f}")

        all_words = set()
        for text in df[self.text_column].values:
            all_words.update(str(text).split())
        print(f"Unique words: {len(all_words)}")

        for col in [self.label_column_1, self.label_column_2]:
            counts = df[col].value_counts()
            print(f"\n{col}:")
            print(f"  Classes: {len(counts)}")
            print(f"  Std dev of class sizes: {counts.std():.1f}")
            print(f"  Min class size: {counts.min()}")
            print(f"  Max class size: {counts.max()}")
