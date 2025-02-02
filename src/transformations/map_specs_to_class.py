import os
import pandas as pd


def map_images_to_classes(image_folder, df, association_column, target_column):
    """
    Maps image files in a folder to their respective target classes using an association between
    an identifier (association column) and a target column.

    Parameters:
        image_folder (str): Path to the folder containing image files (e.g., "../data/raw/1000_mel_spec_seg").
        df (pd.DataFrame): DataFrame containing association and target columns.
        association_column (str): The column name in `df` representing the association class (e.g., "track_id").
        target_column (str): The column name in `df` representing the target class/label (e.g., "danceability_cluster").

    Returns:
        pd.DataFrame: A DataFrame mapping image paths to their target classes.
    """
    # Ensure the specified columns exist in the DataFrame
    if association_column not in df.columns:
        raise KeyError(f"Column '{association_column}' not found in the DataFrame.")
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found in the DataFrame.")

    # Extract the association and target columns as Series
    association_class = df[association_column]
    target_class = df[target_column]

    # Combine the association and target class into a dictionary for faster lookup
    association_to_target = dict(zip(association_class, target_class))

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Create a list to store rows for the resulting DataFrame
    image_data = []

    # Loop through the image files
    #0ahmam4Hqa4hZD1QbUop13_segment_1
    for image_file in image_files:
        # Extract the association ID from the filename
        
        #association_id ="_".join(image_file.split("_")[:-2])  # Adjust based on filename format
        association_id = image_file.split("_")[0]

        # Check if the association ID exists in the mapping
        if association_id in association_to_target:
            # Get the corresponding target class for the association ID
            target_label = association_to_target[association_id]

            # Add the image path and its target class to the list
            image_data.append([os.path.join(image_folder, image_file), target_label])
        else:
            print(f"Association ID {association_id} not found in the association mapping!")

    # Convert the list to a DataFrame and return it
    return pd.DataFrame(image_data, columns=["image_path", "class"])

# run the function
#0ahmam4Hqa4hZD1QbUop13_segment_1
image_folder = "/work3/s222948/data/raw/1000dataset_10seg/specs"
df_path = "/work3/s222948/data/raw/1000dataset.csv"
association_column = "track_id"
target_column = "track_genre"

image_class_mapping = map_images_to_classes(image_folder, pd.read_csv(df_path), association_column, target_column)

# save the image_class_mapping to a CSV file 
image_class_mapping.to_csv("/work3/s222948/data/raw/image_class_mapping.csv", index=False)