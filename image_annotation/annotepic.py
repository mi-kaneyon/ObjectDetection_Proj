import glob 
import os
import numpy as np
from PIL import Image, ImageOps

def process_image_with_padding(input_path, output_folder, target_size=(512, 512), augment=True, padding_ratio=0.1):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image
    img = Image.open(input_path).convert("RGBA")
    
    # Remove the alpha channel (transparency)
    data = np.array(img)
    alpha_data = data[:,:,3]
    non_transparent = alpha_data > 0
    
    # Crop the image to remove transparent areas
    non_transparent_points = np.column_stack(np.where(non_transparent))
    min_coords = non_transparent_points.min(axis=0)
    max_coords = non_transparent_points.max(axis=0)
    cropped_img = Image.fromarray(data[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, :])
    
    # Calculate padding
    width, height = cropped_img.size
    padding_width = int(width * padding_ratio)
    padding_height = int(height * padding_ratio)
    
    # Add padding
    padded_img = ImageOps.expand(cropped_img, (padding_width, padding_height, padding_width, padding_height), fill=(0, 0, 0, 0))
    
    # Resize the padded image while maintaining aspect ratio
    padded_img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create a new blank image with target size
    new_img = Image.new("RGBA", target_size, (0, 0, 0, 0))
    
    # Paste the resized image onto the center of the new blank image
    paste_position = ((target_size[0] - padded_img.size[0]) // 2, 
                      (target_size[1] - padded_img.size[1]) // 2)
    new_img.paste(padded_img, paste_position)
    
    # Save the processed image
    base_name = os.path.basename(input_path)
    processed_output_path = os.path.join(output_folder, f"Processed_{base_name}")
    new_img.save(processed_output_path)
    
    if augment:
        # Perform image augmentation by rotating the image at 45-degree intervals
        for angle in [45, 90, 135, 180, 225, 270, 315]:
            rotated_img = new_img.rotate(angle)
            rotated_output_path = os.path.join(output_folder, f"P{angle}-{base_name}")
            rotated_img.save(rotated_output_path)


# Example usage
output_folder = "images/output"  # Replace with your directory
input_folder = "images"  # Replace with your directory

# Automatically get all PNG files in the directory
for input_path in glob.glob(f"{input_folder}/*.png"):
    process_image_with_padding(input_path, output_folder)
