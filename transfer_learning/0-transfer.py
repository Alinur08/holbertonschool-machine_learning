#!/usr/bin/env python3
"""
0-transfer.py

Trains a Convolutional Neural Network on the CIFAR-10 dataset using
transfer learning with Keras Applications (DenseNet121).
"""

import tensorflow as tensorflow
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes CIFAR-10 data for DenseNet121.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) - CIFAR-10 images.
        Y: numpy.ndarray of shape (m, 1) or (m,) - CIFAR-10 labels.

    Returns:
        X_p: numpy.ndarray - preprocessed images.
        Y_p: numpy.ndarray - one-hot encoded labels.
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def train_cifar10():
    """
    Trains a transfer learning model on CIFAR-10 using pre-computed feature
    extraction to accelerate training, then saves the compiled model to cifar10.h5.
    """
    # 1. Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # 2. Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Input shape for CIFAR-10 images
    input_shape = (32, 32, 3)
    target_size = (224, 224)

    # 3. Define the base model for feature extraction
    # Lambda layer to resize images to 224x224 required by DenseNet121
    input_layer = K.layers.Input(shape=input_shape)
    resized_input = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image,
            height_factor=target_size[0] // input_shape[0],
            width_factor=target_size[1] // input_shape[1],
            data_format="channels_last",
            interpolation="bilinear"
        )
    )(input_layer)

    base_model = K.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=resized_input,
        pooling='avg'
    )

    # Freeze base model weights
    for layer in base_model.layers:
        layer.trainable = False

    # 4. Pre-compute features from frozen layers to save training time
    print("Pre-computing feature representations with frozen DenseNet121...")
    feature_extractor = K.Model(inputs=input_layer, outputs=base_model.output)
    
    train_features = feature_extractor.predict(X_train_p, batch_size=64, verbose=1)
    test_features = feature_extractor.predict(X_test_p, batch_size=64, verbose=1)

    # 5. Build top classification layers
    top_input = K.layers.Input(shape=(base_model.output_shape[1],))
    x = K.layers.Dense(512, activation='relu')(top_input)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    top_model = K.Model(inputs=top_input, outputs=outputs)

    # Compile the top head
    top_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Train the top head using pre-computed features
    callbacks = [
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=2, verbose=1
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
        )
    ]

    print("Training classification head...")
    top_model.fit(
        train_features,
        Y_train_p,
        epochs=15,
        batch_size=64,
        validation_data=(test_features, Y_test_p),
        callbacks=callbacks
    )

    # 7. Assemble end-to-end model and save
    final_output = top_model(base_model.output)
    full_model = K.Model(inputs=input_layer, outputs=final_output)

    full_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    full_model.save('cifar10.h5')
    print("Model saved to cifar10.h5 successfully.")


if __name__ == '__main__':
    train_cifar10()
