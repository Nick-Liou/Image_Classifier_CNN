    
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from keras.callbacks import EarlyStopping
from keras import layers, models
from sklearn.model_selection import train_test_split

from edav import *

def main() -> None:

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


    ####### DEBUG #######
    # Print the shape of the datasets


    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.05, random_state=42, stratify=train_labels)
    
    if debug :
        print("Shape of training images:", train_images.shape)
        print("Shape of training labels:", train_labels.shape)
        print("Shape of testing images:", test_images.shape)
        print("Shape of testing labels:", test_labels.shape)
        print("Shape of validation images:", val_images.shape)
        print("Shape of validation labels:", val_labels.shape)


    if show_edav : 
        # Plot histograms
        plot_histogram(train_labels , class_names , "train")
        plot_histogram(val_labels , class_names , "validation")
        plot_histogram(test_labels , class_names , "test")

        plot_random_images(train_images  ,train_labels  , class_names , 4 )


    return


    # Δημιουργία του CNN μοντέλου
    model = models.Sequential()

    # 1ο Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # 2ο Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # 3ο Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten και Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Τελικό επίπεδο ταξινόμησης
    model.add(layers.Dense(10, activation='softmax'))

    
    print("CNN topology setup completed.")

    # Περίληψη του μοντέλου
    model.summary()

    # Σύνθεση του μοντέλου
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # 4. Ορισμός EarlyStopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 5. Εκπαίδευση του μοντέλου με validation set και early stopping
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

