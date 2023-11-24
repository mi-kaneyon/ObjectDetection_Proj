import os
import shutil

def copy_images(base_dir, filenames, duplicates):
    for filename in filenames:
        # Split the filename from its extension
        basename, ext = os.path.splitext(filename)
        # Copy the file with new names
        for i in range(1, duplicates + 1):
            new_filename = f"{basename}_{str(i).zfill(5)}{ext}"
            original_file = os.path.join(base_dir, filename)
            new_file = os.path.join(base_dir, new_filename)
            shutil.copy(original_file, new_file)
            print(f"Copied: {new_file}")

# Example usage
base_directory = './akira'  # Adjust this path as necessary
original_images = ['akira_front.png', 'akira_sidegl.png', 'akira_top.png', 'akira_rear.png', 'akira_sidem.png']
duplicates_per_image = 16000  # Adjust the number of duplicates as necessary

# Call the function to start copying images
copy_images(base_directory, original_images, duplicates_per_image)

# It is padding data file tool for training data shortage time.
# Manyan3 :)
