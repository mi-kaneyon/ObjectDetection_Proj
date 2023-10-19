import json

def resize_annotations(input_json_path, output_json_path, new_width, new_height):
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    for image_info in data['images']:
        old_width = image_info['width']
        old_height = image_info['height']
        
        x_scale = new_width / old_width
        y_scale = new_height / old_height
        
        image_info['width'] = new_width
        image_info['height'] = new_height
        
        # Update annotations related to this image
        for annotation in [anno for anno in data['annotations'] if anno['image_id'] == image_info['id']]:
            # Rescale bounding box
            x, y, box_w, box_h = annotation['bbox']
            annotation['bbox'] = [x * x_scale, y * y_scale, box_w * x_scale, box_h * y_scale]
            
            # Rescale area
            annotation['area'] *= (x_scale * y_scale)
            
            # Rescale segmentation if it exists
            if 'segmentation' in annotation:
                for seg in annotation['segmentation']:
                    for i in range(0, len(seg), 2):
                        seg[i] *= x_scale
                        seg[i + 1] *= y_scale
    
    # Save the resized annotations
    with open(output_json_path, 'w') as f:
        json.dump(data, f)

# Example usage
input_json_path = "path/to/original_annotations.json"
output_json_path = "path/to/resized_annotations.json"
new_width = 512
new_height = 512

resize_annotations(input_json_path, output_json_path, new_width, new_height)
