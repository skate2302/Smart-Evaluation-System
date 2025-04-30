import os
import io
from google.cloud import vision
from pdf2image import convert_from_path
from PIL import Image

# Initialize Google Vision API client
client = vision.ImageAnnotatorClient()

def extract_text_from_image(image_path):
    """Extract handwritten text from an image using Google Vision API"""
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    # Extract text line by line
    extracted_lines = [block.description for block in response.text_annotations]
    return "\n".join(extracted_lines) if extracted_lines else "No text detected"

def extract_text_from_pdf(pdf_path):
    """Convert each page of a PDF to an image and extract text"""
    images = convert_from_path(pdf_path)
    full_text = ""

    for i, image in enumerate(images):
        print(f"\n🔍 Processing Page {i+1}...\n")
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        vision_image = vision.Image(content=img_byte_arr)
        response = client.document_text_detection(image=vision_image)

        # Extract text line by line
        extracted_lines = [block.description for block in response.text_annotations]
        full_text += "\n".join(extracted_lines) + "\n\n" if extracted_lines else "No text detected"

    return full_text.strip()

# File Path (Change this to your file)
file_path = "Image_to_PDF_(Handwriting).pdf"  # or .jpg, .png

# Check file type and process accordingly
if file_path.lower().endswith(".pdf"):
    extracted_text = extract_text_from_pdf(file_path)
else:
    extracted_text = extract_text_from_image(file_path)

# Print extracted text line by line
print("\n📝 Extracted Handwritten Text:\n")
print(extracted_text)

# import os
# print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
