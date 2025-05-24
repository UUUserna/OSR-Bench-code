import os
import json
from collections import defaultdict

def main(prefixes=None):
    # Path to the data directory
    data_dir = 'data/'
    if prefixes:
        output_file = os.path.join(data_dir, f'negative_sampling_{"_".join(prefixes)}.json')
    else:
        output_file = os.path.join(data_dir, 'negative_sampling_all.json')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Dictionaries to store our results
    object_frequency = defaultdict(int)  # To count in how many images each object appears
    object_cooccurrence = defaultdict(lambda: defaultdict(int))  # To count co-occurrences
    total_images = 0  # Counter for total number of processed images

    # Walk through all subdirectories in data_dir
    for root, dirs, files in os.walk(data_dir):
        # Check if the directory name matches any of the prefixes
        if prefixes and not any(root.startswith(os.path.join(data_dir, prefix)) for prefix in prefixes):
            continue

        # Check if cognitive_map.json exists in the current directory
        if 'cognitive_map.json' in files:
            # Full path to the json file
            json_path = os.path.join(root, 'cognitive_map.json')
            
            try:
                # Load the json file
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract class_count information
                if 'class_count' in data:
                    class_count = data['class_count']
                else:
                    class_count = data  # Assume the whole file is class_count
                
                # Determine the format and extract objects
                objects_in_image = []
                if isinstance(class_count, dict):
                    objects_in_image = [obj for obj, count in class_count.items() if count > 0]
                elif isinstance(class_count, list):
                    # If it's a list of object names
                    if all(isinstance(item, str) for item in class_count):
                        objects_in_image = class_count
                    # If it's a list of objects with some property indicating presence
                    elif all(isinstance(item, dict) for item in class_count):
                        # This would need to be adapted based on the actual structure
                        # Example: [{name: 'dog', count: 2}, {name: 'cat', count: 1}]
                        for item in class_count:
                            if 'name' in item and item.get('count', 0) > 0:
                                objects_in_image.append(item['name'])
                            elif 'class' in item and item.get('count', 0) > 0:
                                objects_in_image.append(item['class'])
                
                # If we couldn't determine objects, skip this image
                if not objects_in_image:
                    print(f"No objects found in {json_path}")
                    continue
                
                total_images += 1  # Count only successfully processed images
                
                # Update object frequency
                for obj in objects_in_image:
                    object_frequency[obj] += 1
                
                # Update co-occurrence counts
                for i, obj1 in enumerate(objects_in_image):
                    for obj2 in objects_in_image[i+1:]:  # Avoid counting pairs twice
                        object_cooccurrence[obj1][obj2] += 1
                        object_cooccurrence[obj2][obj1] += 1  # Symmetric relation
            
            except Exception as e:
                print(f"Error processing {json_path}: {e}")

# # Calculate frequencies and co-occurrence rates
# object_frequency_rate = {obj: count / total_images if total_images > 0 else 0 
#                         for obj, count in object_frequency.items()}

# cooccurrence_rate = {}
# for obj1 in object_frequency:
#     cooccurrence_rate[obj1] = {}
#     for obj2 in object_frequency:
#         if obj1 != obj2:
#             # Calculate as proportion of images where obj1 appears
#             if object_frequency[obj1] > 0:
#                 cooccurrence_rate[obj1][obj2] = object_cooccurrence[obj1][obj2] / object_frequency[obj1]
#             else:
#                 cooccurrence_rate[obj1][obj2] = 0

    # Prepare output
    results = {
        'object_frequency': dict(object_frequency),
        # 'object_frequency_rate': object_frequency_rate,
        'cooccurrence_count': dict(object_cooccurrence),
        # 'cooccurrence_rate': cooccurrence_rate
    }

    # Save results to output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    prefixes = sys.argv[1:] if len(sys.argv) > 1 else None
    main(prefixes)
    