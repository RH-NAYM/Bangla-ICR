from PIL import Image
import os
import json
import cv2
import chardet  # Install with: pip install chardet

# Define dataset paths
RAW_DIR = 'raw'
OUTPUT_DIR = 'img'
TEMP_FILE = "temp.jpg"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get sorted file list
file_arr = sorted(os.listdir(RAW_DIR))  # Sorting ensures consistency
failed_to_save = 0

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read a small portion of the file
    result = chardet.detect(raw_data)
    return result['encoding']

# Loop through files in pairs
for file_number in range(0, len(file_arr) - 1, 2):  
    try:
        img_name = file_arr[file_number]
        img_json = file_arr[file_number + 1]

        # Ensure correct file extensions
        if not img_json.endswith('.json') or not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        print(f"Processing: {img_name}, {img_json}")

        json_path = os.path.join(RAW_DIR, img_json)

        # Detect encoding before reading
        encoding = detect_encoding(json_path)
        if not encoding:
            print(f"Skipping {json_path}: Unable to detect encoding.")
            continue

        # Read JSON with detected encoding
        try:
            with open(json_path, "r", encoding=encoding) as read_it:
                data = json.load(read_it)
        except UnicodeDecodeError:
            print(f"Skipping {json_path}: Unicode error with detected encoding {encoding}.")
            continue
        except json.JSONDecodeError:
            print(f"Skipping {json_path}: Invalid JSON format.")
            continue

        # Open the image
        img_path = os.path.join(RAW_DIR, img_name)
        im = Image.open(img_path)

        for item in data.get('shapes', []):
            try:
                x, y = item['points'][0]
                x1, y1 = item['points'][1]
                img_label = item['label'].strip().replace('*', 'star').replace('/', '_')  # Sanitize filenames

                # Crop image
                cropped_img = im.crop((x, y, x1, y1))

                # Convert to binary using OTSU Thresholding
                cropped_img.save(TEMP_FILE)
                img = cv2.imread(TEMP_FILE, 0)
                _, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                im_bw = Image.fromarray(imgf)

                # Construct output filename
                sanitized_label = img_label if len(img_label) < 16 else img_label[:15]
                output_filename = f"{sanitized_label} {file_number}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                # Save processed image
                im_bw.save(output_path)
                print(f"Saved: {output_filename}")

            except Exception as e:
                print(f"Error processing word: {item.get('label', 'UNKNOWN')} - {e}")
                failed_to_save += 1

    except IndexError:
        print(f"Skipping unmatched file: {file_arr[file_number]}")
    except Exception as e:
        print(f"Unexpected error processing {file_arr[file_number]}: {e}")

# Cleanup temp file
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

print(f"Total failed to save: {failed_to_save}")
