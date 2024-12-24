from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(observations: np.ndarray, names: List[str] , data_name: str = "train" , save_folder: str = "") -> None:
    """
    Creates a histogram with labeled bars and a y-axis for frequency percentage.
    
    Args:
    observations (np.ndarray): A list of integers representing observations (raw data).
    names (List[str]): A list of strings representing category names for the observations.

    Raises:
    ValueError: If the observations contain values outside the range of the provided names.
    """
    # Use np.unique to get the unique values and their frequencies
    unique_values, counts = np.unique(observations, return_counts=True)

    # Check if all observations are valid indices for names
    if not np.all(np.isin(unique_values, np.arange(len(names)))):
        raise ValueError("Observations contain values outside the range of category names.")
    
    # Calculate total number of observations for frequency percentage
    total_count = sum(counts)
    
    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # x-axis positions for the bars
    x_positions = np.arange(len(counts))

    # Calculate frequencies as percentage
    frequencies = [(i / total_count) * 100 for i in counts]

    # Plot the histogram bars
    bars = ax1.bar(x_positions, frequencies, color='skyblue')
    
    # Set labels for the x-axis and y-axis (only frequency percentage on y-axis)    
    plt.title(f"Frequencies of {data_name} data")
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Frequency (%)', color='black')
    
    # Scale the y-axis for percentage values
    # ax1.set_ylim(0, 100)  # Set the y-axis limit from 0 to 100%
    
    # Add the count labels on top of each bar (absolute frequency)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}', ha='center', va='bottom', color='blue')
    
    # Set the ticks and labels for the x-axis
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([names[val] for val in unique_values])

    # Show the plot
    plt.tight_layout()

    if save_folder != "" :
        # Save to the specified folder
        plt.savefig(f"{save_folder}/plot_hist_{data_name.replace(' ','_').replace('/','_')}.eps", format='eps')
    
    plt.show()


def plot_random_images(train_images: np.ndarray, train_labels: np.ndarray, class_names: List[str], im_per_class: int = 4 , save_folder: str = "" , label_origin: str = "data")  -> None:
    """
    Plots a grid of random images for each class from the training dataset.

    Args:
        train_images (np.ndarray): Array of training images. Expected shape is (num_samples, height, width, channels).
        train_labels (np.ndarray): Array of training labels. Expected shape is (num_samples, 1) or (num_samples,) or similar.
        class_names (List[str]): List of class names corresponding to the class indices.
        im_per_class (int): Number of random images to display per class. Defaults to 4.

    Returns:
        None: This function displays the images using matplotlib but returns no values.
    """
    
    # Number of classes based on the length of class_names
    num_classes = len(class_names)

    train_labels = train_labels.squeeze()

    # Create a figure with subplots (rows = im_per_class, columns = num_classes)
    fig, axes = plt.subplots(im_per_class, num_classes, figsize=(13, 7))

    # Adjust the layout to remove extra spaces between subplots
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # Loop through each class
    for class_idx in range(num_classes):
        # Get all images corresponding to the current class
        class_images = train_images[train_labels == class_idx]

        # Randomly select `im_per_class` images from the class
        random_indices = np.random.choice(len(class_images), im_per_class, replace=False)

        # Plot the randomly selected images for the current class
        if im_per_class == 1 :
            # Display the selected image in the corresponding subplot
            axes[class_idx].imshow(class_images[0])

            # Set title to the class name 
            axes[class_idx].set_title(class_names[class_idx] )

            # Hide axis ticks and labels for a cleaner visualization
            axes[class_idx].axis('off')
        else:            
            for img_idx, ax_row in zip(random_indices, range(im_per_class)):
                # Display the selected image in the corresponding subplot
                axes[ax_row, class_idx].imshow(class_images[img_idx])
                
                # Set title to the class name
                axes[ax_row, class_idx].set_title(class_names[class_idx])
                # Set title to the class name only for the first image in each column (class)
                # axes[ax_row, class_idx].set_title(class_names[class_idx] if ax_row == 0 else "")

                # Hide axis ticks and labels for a cleaner visualization
                axes[ax_row, class_idx].axis('off')
            

    # Apply tight layout to ensure the subplots fit well in the figure area
    plt.tight_layout()
    
    if save_folder != "" :
        # Save to the specified folder    
        file_name = f"{save_folder}/plot_radom_images_{label_origin}.eps"
        plt.savefig(file_name, format='eps')

    # Show the figure
    plt.show()

    


if __name__ == "__main__":
    # Example usage:
    observations = np.array([0, 1, 1, 2, 2, 2, 3, 1, 0, 0, 3, 1, 2, 2])
    names = ['Category A', 'Category B', 'Category C', 'Category D']
    plot_histogram(observations, names)
