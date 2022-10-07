import tensorflow as tf


max_features = 1000
sequence_length = 500
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def fit_predict(raw_train_ds, raw_val_ds):
    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Vectorized text", vectorize_text(first_review, first_label))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim = 16

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features + 1, embedding_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=['accuracy']
    )

    return history, export_model
