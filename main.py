    
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping
from keras import layers, models
import keras
from sklearn.model_selection import StratifiedKFold, train_test_split

from typing import Any

from edav import *


def create_model(classifier_id : int = 0 ) -> tf.keras.Model:
    model:tf.keras.Model = models.Sequential()
        
    # Define the input layer first
    # model.add(layers.InputLayer(shape=(32, 32, 3)))

    # 1st Convolutional Block    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # 2nd Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # 3rd Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten and then add Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

	# Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # "sparse_categorical_crossentropy"

    return model


def main(classifier_id : int = 0) -> None:

    show_edav = True
    debug = True

    # Load the CIFAR-10 dataset
    test_images: np.ndarray
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    print("Dataset download completed")

    # CIFAR-10 class labels
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    assert train_images.shape == (50000, 32, 32, 3)
    assert test_images.shape == (10000, 32, 32, 3)
    assert train_labels.shape == (50000, 1)
    assert test_labels.shape == (10000, 1)



    # train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.05, random_state=42, stratify=train_labels)
    
    if debug :
        print("Shape of training images:", train_images.shape)
        print("Shape of training labels:", train_labels.shape)
        print("Shape of testing images:", test_images.shape)
        print("Shape of testing labels:", test_labels.shape)
        # print("Shape of validation images:", val_images.shape)
        # print("Shape of validation labels:", val_labels.shape)


    if show_edav : 
        # Plot histograms
        plot_histogram(train_labels , class_names , "train")
        # plot_histogram(val_labels , class_names , "validation")
        plot_histogram(test_labels , class_names , "test")

        plot_random_images(train_images  ,train_labels  , class_names , 4 )



    # # Normalization
    x_train = train_images.astype('float32') / 255
    x_test = test_images.astype('float32') / 255

    # One-Hot encoding
    n_classes = 10
    y_train = keras.utils.to_categorical(train_labels, n_classes)
    y_test = keras.utils.to_categorical(test_labels, n_classes)
    # y_val =  keras.utils.to_categorical(val_labels, n_classes)    
    print("Normalization and one-hot encoding completed")

    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=['Classifier Name', 'Set Type', 'Number of samples', 'Accuracy', 'Precision', 'Recall', 'F1-score'])


	# Define EarlyStopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Set parameters
    num_folds = 5

    
    # Stratified k-fold split
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(skf.split(x_train, train_labels)):

        print(f"<================ Starting fold {fold + 1}/{num_folds} ================>")

        x_train_fold, y_train_fold = x_train[train_indices], y_train[train_indices]
        x_val_fold, y_val_fold = x_train[val_indices], y_train[val_indices]

        if show_edav:            
            plot_histogram(np.argmax(y_train_fold, axis=1), class_names , f"fold {fold + 1}/{num_folds} train")
            plot_histogram(np.argmax(y_val_fold, axis=1) , class_names , f"fold {fold + 1}/{num_folds} validation")
            
            
        model = create_model(classifier_id)



    return
    


	# Train the model
    print("Starting model training...")
    history = model.fit(
        train_images, train_labels, 
        epochs=50, 
        validation_data=(val_images, val_labels), 
        callbacks=[early_stopping]
    )

    print("Training completed.")

    # 6. Εμφάνιση των loss curves (training & validation)
    print("Plotting loss curves...")
    plt.figure(figsize=(12, 4))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()



if __name__ == "__main__":

    
    # # Check if TensorFlow detects the GPU
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # # Check GPU device name and confirm that cuDNN is being used
    # gpu_devices = tf.config.list_physical_devices('GPU')
    # if gpu_devices:
    #     for gpu in gpu_devices:
    #         print(f"Device Name: {gpu.name}")
    #         print(f"Device Type: {gpu.device_type}")


    main()

