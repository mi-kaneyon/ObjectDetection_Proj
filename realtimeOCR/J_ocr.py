# -*- coding: utf-8 -*-
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_text_on_frame(frame, text, position=(50, 50), font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", font_size=24):
    # Convert the OpenCV rectangle to a PIL image
    cv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_frame)

    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text
    draw.text(position, text, font=font, fill=(255,255,255,255))

    # Convert back to OpenCV image and return
    cv_frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_frame_with_text

def realtime_ocr_with_text_display():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect text in the frame
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(img_pil, lang='jpn')

        # Draw the detected text on the frame
        frame_with_text = draw_text_on_frame(frame, text, position=(50, 50), font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", font_size=24)

        # Display the frame
        cv2.imshow('Frame with Text', frame_with_text)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_ocr_with_text_display()
