from PIL import Image
import os

def crop_center_and_resize(img, final_size):
    width, height = img.size

    left_margin = (width - min(width, height)) / 2
    top_margin = (height - min(width, height)) / 2
    right_margin = (width + min(width, height)) / 2
    bottom_margin = (height + min(width, height)) / 2

    img = img.crop((left_margin, top_margin, right_margin, bottom_margin))
    img = img.resize((final_size, final_size), Image.ANTIALIAS)
    return img

def process_images(input_folder, output_folder, final_size=512):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(input_folder, filename)).convert("RGBA")
            img = crop_center_and_resize(img, final_size)
            img.save(os.path.join(output_folder, filename))

# Uncomment and specify your input and output folder paths
input_folder = "path/to/your/input/folder"
output_folder = "path/to/your/output/folder"

# process_images(input_folder, output_folder)
