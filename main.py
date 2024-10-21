    
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def main() -> None:

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


if __name__ == "__main__":

    main()

