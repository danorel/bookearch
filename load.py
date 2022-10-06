import tensorflow as tf


def load_dataset(directory: str):
    batch_size = 32
    seed = 42
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        directory,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        directory,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        directory,
        batch_size=batch_size)
    return raw_train_ds, raw_val_ds, raw_test_ds
