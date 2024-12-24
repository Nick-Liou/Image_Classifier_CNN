    
# from google.colab import drive

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import os

from keras.callbacks import EarlyStopping
from keras import layers, models
import keras
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from typing import Any

from edav import *


def create_model(classifier_id : int = 0 ) -> tf.keras.Model:
    model:tf.keras.Model = models.Sequential()
    input_shape = (32, 32, 3)

    # Set learning rate and regularization
    # learning_rate = 0.001   # Try 0.001, 0.01, 0.005, 0.0001, 0.00001
    # weight_decay = 0.01     # Try 0.004
    # optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)   # Try adam, SGD

    # Add the Input layer with the specified shape
    model.add(layers.Input(shape=input_shape))

    if classifier_id == 0:  # Model 1
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # Flatten the output of the convolutional layers
        model.add(layers.Flatten())

        # Fully connected layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    elif classifier_id == 1:  # Model 2
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # Flatten the output of the convolutional layers
        model.add(layers.Flatten())

        # Fully connected layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

        # Compile the model, changed loss function to sparse categorical cross entropy
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    elif classifier_id == 2:  # Model 3
        # Convolutional Layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        # Flatten
        model.add(layers.Flatten())

        # Fully connected layers
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    else:
        raise ValueError("Invalid classifier_id.")

    return model



def main(classifier_id : int = 0) -> None:

    save_model = False
    show_edav = True
    debug = True
    local_run = True

    if local_run :
        model_save_folder = "Trained_models"
    else:
        # Mount Google Drive
        drive.mount('/content/drive')
        print("Google Drive mounted succesfully")

        # Specify Google Drive path
        model_save_folder = "/content/drive/MyDrive/Trained_models"

    os.makedirs(model_save_folder, exist_ok=True)  # Create folder if it doesn't exist    
    print(f"Model save folder: {model_save_folder}")
    

    # Set to "" to not save plots
    plot_save_folder = f"Saved_Plots_from_model_{classifier_id}"
    plot_save_folder = ""
    if plot_save_folder != "":
        os.makedirs(plot_save_folder, exist_ok=True)  # Create folder if it doesn't exist


    # from keras.models import load_model
    # # Load the model from the .h5 file
    # loaded_model = load_model('my_model.h5')

    

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

    print(f"We have {len(train_images)} training examples of size {train_images.shape[1:3]} and {len(test_images)} testing examples of size {test_images.shape[1:3]}")
    print(f"Dataset labels: {class_names}")

    if debug :
        print("Shape of training images:", train_images.shape)
        print("Shape of training labels:", train_labels.shape)
        print("Shape of testing images:", test_images.shape)
        print("Shape of testing labels:", test_labels.shape)
        


    if show_edav : 
        # Plot histograms
        plot_histogram(train_labels , class_names , "train", save_folder=plot_save_folder)
        plot_histogram(test_labels , class_names , "test", save_folder=plot_save_folder)

        plot_random_images(train_images  ,train_labels  , class_names , 4 , save_folder=plot_save_folder )



    # Normalization
    x_train = train_images.astype('float32') / 255
    x_test = test_images.astype('float32') / 255

    # One-Hot encoding
    n_classes = 10
    y_train = keras.utils.to_categorical(train_labels, n_classes)
    y_test = keras.utils.to_categorical(test_labels, n_classes)
    print("Normalization and one-hot encoding completed")


    # Initialize a list to store results
    results_list = []
    

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
            plot_histogram(np.argmax(y_train_fold, axis=1), class_names , f"fold {fold + 1}/{num_folds} train" , save_folder= plot_save_folder)
            plot_histogram(np.argmax(y_val_fold, axis=1) , class_names , f"fold {fold + 1}/{num_folds} validation", save_folder= plot_save_folder)

            
            # Convert y_train_fold to a pandas Series (if it's not already)
            y_series = pd.Series(y_train_fold.argmax(axis=1))

            # Replace numeric labels with class names
            y_named_series = y_series.map(lambda x: class_names[x])

            # Plot with labeled categories
            plt.figure(figsize=(12, 6))
            plt.scatter(y_named_series.index, y_named_series, marker='o', label='Labels', alpha=0.6)
            plt.title('Time Series Plot of y_train_fold with Class Names', fontsize=14)
            plt.xlabel('Time (Index)', fontsize=12)
            plt.ylabel('Class Name', fontsize=12)
            plt.grid(axis='y')
            plt.legend()
            # plt.show()


            # plt.figure(figsize=(12, 6))
            # sns.scatterplot(x=y_named_series.index, y=y_named_series, hue=y_named_series, palette='tab10') #, s=50
            # plt.title('Time Series Plot of y_train_fold with Class Names', fontsize=14)
            # plt.xlabel('Time (Index)', fontsize=12)
            # plt.ylabel('Class Name', fontsize=12)
            # plt.grid(axis='y')
            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.show()
            
            
        model = create_model(classifier_id)


        # Train the model
        print("Starting model training...")
        history = model.fit(x_train_fold, y_train_fold, batch_size=128, epochs=2, validation_data=(x_val_fold, y_val_fold), callbacks=[early_stopping])
        print("Training completed.")


       
        if save_model:
            # Get the current date and time
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Extract the final training and validation accuracy
            train_accuracy = history.history['accuracy'][-1]  # Last epoch training accuracy
            val_accuracy = history.history['val_accuracy'][-1]  # Last epoch validation accuracy
            
            # Format the accuracies as percentages
            train_accuracy = round(train_accuracy * 100, 2)
            val_accuracy = round(val_accuracy * 100, 2)
            
            # Construct the file name with the accuracies
            model_file_name = f"{model_save_folder}/model{classifier_id}_acc{train_accuracy}_valacc{val_accuracy}_f{fold+1}_{current_time}.keras"
            
            print(f"Saving model to: {model_file_name}")
            # Save the model
            model.save(model_file_name)

        
        print(f"Plotting loss curves for Fold {fold + 1}...")

        plt.figure(figsize=(12, 6))

        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss for Fold {fold + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy for Fold {fold + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

        
        
        # Calculate metrics for train, validation, and test sets
        print(f"Metrics for Fold {fold + 1}:")
        for (x , y, set_name) in [[x_train_fold,y_train_fold,"Train"], [x_val_fold,y_val_fold,"Validation"], [x_test,y_test,"Test"] ] :

            y_true = y.argmax(axis=1)
            y_pred = model.predict(x).argmax(axis=1)

            print(f"{set_name} set metrics:")
            train_accuracy = accuracy_score(y_true, y_pred)
            train_precision = precision_score(y_true, y_pred, average='weighted')
            train_recall = recall_score(y_true, y_pred, average='weighted')
            train_f1 = f1_score(y_true, y_pred, average='weighted')
            print(f"Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1-Score: {train_f1}")


            # Collect the metrics for each set in a dictionary
            results_list.append({
                "Classifier Name": f"CNN_Classifier_{classifier_id}",
                "Set Type": set_name,
                "Number of samples": len(x),
                "Accuracy": train_accuracy,
                "Precision": train_precision,
                "Recall": train_recall,
                "F1-score": train_f1
            })

            # Print the confusion matrix for the train, validation, and test sets
            cm = confusion_matrix(y_true, y_pred)

            print(f"Confusion Matrix - {set_name}:")
            print(cm)

            # Plotting the confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(f"Confusion Matrix - {set_name}")
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.show()

            # Prediction Based Image Plots
            if set_name == "Test":                
                plot_random_images(x,y_pred,class_names, save_folder=plot_save_folder , label_origin= f"pred_fold_{fold+1}")



    
    # Create a DataFrame from the list of results
    results_df = pd.DataFrame(results_list)
    # Display the results
    print(results_df)


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

