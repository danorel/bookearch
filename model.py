import tensorflow as tf

max_features = 10000
sequence_length = 250
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_sequence_length=sequence_length)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def fit_predict(raw_train_ds, raw_val_ds, raw_test_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim = 16

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features + 1, embedding_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)])
    model.summary()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    return history
