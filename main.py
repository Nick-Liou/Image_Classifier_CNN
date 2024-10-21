    
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from keras.callbacks import EarlyStopping
from keras import layers, models

def main() -> None:

    show_image = False

    # Load the CIFAR-10 dataset
    test_images: np.ndarray
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    assert train_images.shape == (50000, 32, 32, 3)
    assert test_images.shape == (10000, 32, 32, 3)
    assert train_labels.shape == (50000, 1)
    assert test_labels.shape == (10000, 1)


    # Print the shape of the dataset
    print(f"Training data shape: {train_images.shape}, Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

    from sklearn.model_selection import train_test_split


    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.05, random_state=42, stratify=train_labels)


    print(f"New Training data shape: {train_images.shape}, Training labels shape: {train_labels.shape}")
    print(f"Validation data shape: {val_images.shape}, Validation labels shape: {val_labels.shape}")


    
    # CIFAR-10 class labels
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    if show_image : 
        # Plot 4 random images per class
        fig, axes = plt.subplots(10, 4, figsize=(10, 25))
        # fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        for class_idx in range(10):
            class_images = train_images[train_labels[:, 0] == class_idx]
            random_indices = np.random.choice(len(class_images), 4, replace=False)
            for img_idx, ax in zip(random_indices, axes[class_idx]):
                ax.imshow(class_images[img_idx])
                ax.set_title(class_names[class_idx])
                ax.axis('off')

        plt.show()



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

    
    # Check if TensorFlow detects the GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Check GPU device name and confirm that cuDNN is being used
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            print(f"Device Name: {gpu.name}")
            print(f"Device Type: {gpu.device_type}")





    # main()

