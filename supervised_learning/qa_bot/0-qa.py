#!/usr/bin/env python3

import tensorflow as tf
import transformers
import tensorflow_hub


def question_answer(question, reference):
    """Answer the question based on the contexte (reference)

    Args:
        question (string): question to answer
        reference (string): contexte where to find the answer

    Returns:
        string: answer of the question
    """
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = tensorflow_hub.load(
        "https://www.kaggle.com/models/seesee/bert/TensorFlow2/uncased-tf2-qa/1"
    )

    question_tokens = tokenizer.tokenize(question)
    para_tokens = tokenizer.tokenize(reference)

    tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + para_tokens + ["[SEP]"]

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_word_ids)
    # Créer une liste d'id pour la question (0) et le
    # contexte (1) les + 1 ou 1 + ajoute les séparateurs [CLS] & [SEP] * 2
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (
        len(para_tokens) + 1
    )

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids),
    )

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start:short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
