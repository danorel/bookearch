import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


def load_dataset_data(directory: str, filename: str = "data.csv"):
    names = ['title', 'genre', 'summary']
    dtype = {'title': str, 'genre': str, 'summary': str}
    df = pd.read_csv(
        os.path.join(directory, filename),
        header=None,
        names=names,
        dtype=dtype
    )
    df.info()
    train_df = df.sample(frac=0.8, random_state=34)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def load_dataset(directory: str):
    train_df, test_df = load_dataset_data(directory)
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        train_df,
        train_df["genre"],
        num_epochs=None,
        shuffle=True
    )
    test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        test_df,
        test_df["genre"],
        shuffle=False
    )
    return train_input_fn, test_input_fn


def fit(train_input_fn):
    embedded_text_feature_column = hub.text_embedding_column(
        key="summary",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=5,
        optimizer=tf.keras.optimizers.Adagrad(lr=0.003))
    estimator.train(input_fn=train_input_fn, steps=5000)
    train_eval_result = estimator.evaluate(input_fn=train_input_fn)
    print("Training set accuracy:{accuracy}".format(**train_eval_result))
    return estimator


def predict(estimator, test_input_fn):
    test_eval_result = estimator.evaluate(input_fn=test_input_fn)
    print("Testing set accuracy:{accuracy}".format(**test_eval_result))
