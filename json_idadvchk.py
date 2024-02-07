import json
from collections import Counter

def main():
    # Load the annotations
    with open('coco_annotations.json', 'r') as f:
        annotations = json.load(f)
    
    # Preprocess to create a mapping from image ID to file name and category names
    image_id_to_info = {img['id']: img['file_name'] for img in annotations['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in annotations['categories']}

    # Initialize counters
    label_counter = Counter()
    prefix_counter = Counter()
    checked_images = 0

    # Check for the first 30 images if they are registered correctly
    for annotation in annotations['annotations']:
        if checked_images >= 30:
            break  # Stop after checking the first 30 images
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if category_id < 4:  # Checking only for labels 0 to 3
            file_name = image_id_to_info.get(image_id)
            if file_name:
                prefix = file_name.split('_')[0]
                prefix_counter[prefix] += 1
                label_counter[category_id] += 1
                checked_images += 1

    # Continue counting for the rest of the images
    for annotation in annotations['annotations'][checked_images:]:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        label_counter[category_id] += 1
        if image_id in image_id_to_info:  # Ensure the image ID is in the mapping
            file_name = image_id_to_info[image_id]
            prefix = file_name.split('_')[0]
            prefix_counter[prefix] += 1

    # Display the count results
    print_counts(label_counter, category_id_to_name)
    print(f"Prefix counts: {dict(prefix_counter)}")

def print_counts(counter, category_mapping):
    for label, count in counter.items():
        label_name = category_mapping.get(label, 'Unknown label')
        print(f"Label {label} ({label_name}) has {count} instances")

if __name__ == "__main__":
    main()
