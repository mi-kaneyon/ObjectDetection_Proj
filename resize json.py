import json

def resize_annotations(input_json, output_json, new_width, new_height):
    # Load the existing annotations
    with open(input_json, 'r') as f:
        annotations = json.load(f)
    
    # Get the original image dimensions from the first annotation
    # Assuming all images have the same dimensions
    original_width = annotations['images'][0]['width']
    original_height = annotations['images'][0]['height']
    
    # Calculate the scaling factors
    width_scale = new_width / original_width
    height_scale = new_height / original_height
    
    # Update the image dimensions in the annotations
    for image in annotations['images']:
        image['width'] = new_width
        image['height'] = new_height
    
    # Scale the bounding box coordinates
    for annotation in annotations['annotations']:
        x, y, w, h = annotation['bbox']
        annotation['bbox'] = [
            x * width_scale,
            y * height_scale,
            w * width_scale,
            h * height_scale
        ]
    
    # Save the updated annotations to a new file
    with open(output_json, 'w') as f:
        json.dump(annotations, f, indent=2)

# Usage:
input_json = 'path/to/original_annotations.json'
output_json = 'path/to/resized_annotations.json'
new_width = 640
new_height = 640
resize_annotations(input_json, output_json, new_width, new_height)
