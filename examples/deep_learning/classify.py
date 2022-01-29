import string, re
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.optimizers import Adam
from fastcore.all import run
import numpy as np


def to_np(ds): 
    """Convert tfds.datasets to flat numpy arrays"""
    return [np.stack(d) for d in zip(*tfds.as_numpy(ds.unbatch()))]


def get_data(url='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', bs=32, seed=1234):
    "Code is from https://keras.io/examples/nlp/text_classification_from_scratch/"

    run(f'curl -O {url}')
    run('tar -xf aclImdb_v1.tar.gz')
    run('rm -rf aclImdb/train/unsup')

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=bs,
        validation_split=0.2,
        subset="training",
        seed=seed,
    )
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=bs,
        validation_split=0.2,
        subset="validation",
        seed=seed,
    )
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        "aclImdb/test", batch_size=bs
    )
    
    train, val, test = map(to_np, [raw_train_ds, raw_val_ds, raw_test_ds])
    np_dict = {'train':train, 'val':val, 'test':test}
    return np_dict

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

def build_model(train_x, 
                embedding_dim = 128, 
                max_features=20000, 
                sequence_length=500,
                learning_rate=.01):
    
    # Vectorization layer to process raw strings as part of the model.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorize_layer.adapt(train_x)

    inputs = tf.keras.Input(shape=(1,), dtype="string")
    indices = vectorize_layer(inputs)

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(indices)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def train_model(model, data, num_epochs=1, batch_size=64):
    x_train, y_train = data['train']
    x_val, y_val = data['val']
    x_test, y_test = data['test']
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    history = model.fit(x=x_train, 
                        y=y_train, 
                        validation_data=(x_val, y_val), 
                        epochs=num_epochs,
                        batch_size=batch_size,
                        callbacks=my_callbacks)
    
    val_metrics = model.evaluate(x=x_val, y=y_val)
    test_metrics = model.evaluate(x=x_test, y=y_test)
    
    test_predictions = model.predict(x_test)
    return {'history':history.history, 
            'weights': model.get_weights(),
            'test_inputs':x_test,
            'test_predictions':test_predictions,
            'test_labels':y_test,
            'final_val_metrics': dict(zip(model.metrics_names, val_metrics)),
            'final_test_metrics': dict(zip(model.metrics_names, test_metrics))}


if __name__ == '__main__':
    data = get_data(url='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    model = build_model(data['train'][0])
    artifacts = train_model(model=model, data=data)
