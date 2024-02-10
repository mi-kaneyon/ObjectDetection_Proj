
# -*- coding: utf-8 -*-
import cv2
import pytesseract
from PIL import Image
import numpy as np

def preprocess_frame(frame):
    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def realtime_ocr():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the frame to improve OCR accuracy
        preprocessed_frame = preprocess_frame(frame)

        # Convert frame for pytesseract
        img_pil = Image.fromarray(preprocessed_frame)

        # Detect text in the frame
        text = pytesseract.image_to_string(img_pil, lang='jpn')

        # Display the detected text in the console
        print(text)

        # Show the preprocessed frame (Optional)
        cv2.imshow('Preprocessed Frame', preprocessed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_ocr()
