#!/usr/bin/env python3
"""Dataset module for machine translation using TED HRLR dataset."""

import transformers
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation."""

    def __init__(self):
        """Initialize the Dataset instance.

        Loads the TED HRLR Portuguese to English translation dataset
        and creates tokenizers for both languages.
        """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence pairs.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en) trained tokenizers for
                Portuguese and English respectively.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            """Yield English sentences from the dataset."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
