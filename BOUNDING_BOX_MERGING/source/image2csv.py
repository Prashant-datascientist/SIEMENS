import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

import pandas as pd
import os

# Path to the directory containing the images (update this path)
image_dir = "C:\\Users\prash\Desktop\SIEMENS\SIEMENS_proj_folder\BOUNDING_BOX_MERGING\data\screening_data_csv_1"

# Initialize an empty DataFrame to store extracted data
df = pd.DataFrame(columns=["List A", "List B"])

# Iterate over each image in the directory
for image_file in os.listdir(image_dir):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        
        # Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(image)
        
        # Split the extracted text into lines
        lines = extracted_text.split("\n")
        
        # Assuming the format is [x1, y1, x2, y2]
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                bounding_box_a = lines[i]
                bounding_box_b = lines[i + 1]
                df = df.append({"List A": bounding_box_a, "List B": bounding_box_b}, ignore_index=True)

# Save the extracted data to a CSV file
df.to_csv("extracted_data.csv", index=False)

print(f"Data extracted from {len(df)} images and saved to extracted_data.csv")