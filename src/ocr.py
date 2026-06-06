import os
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

class LicensePlateOCR:
    def __init__(self):
        """
        Initializes the PaddleOCR model.
        Using English language ('en') as it is optimal for alphanumeric character recognition
        without loading heavy language models.
        """
        # Initialize PaddleOCR
        # lang='en' is fast and extremely accurate for alphanumeric characters.
        # We disable oneDNN/MKLDNN on CPU because it has a known bug in recent versions
        # that raises "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support".
        try:
            self.ocr = PaddleOCR(lang='en', use_textline_orientation=True, enable_mkldnn=False)
        except Exception:
            try:
                self.ocr = PaddleOCR(lang='en', use_angle_cls=True, enable_mkldnn=False)
            except Exception:
                self.ocr = PaddleOCR(lang='en')

    def format_plate_text(self, text):
        """
        Cleans and formats license plate text to standard Vietnamese format.
        Example: 51G10096 -> 51G-100.96
        """
        # Convert to uppercase and strip non-alphanumeric except hyphen and dot
        text = re.sub(r'[^A-Z0-9\-\.]', '', text.upper())
        
        # If it's already well-formatted, return it
        if re.match(r'^\d{2}[A-Z]{1,2}\d{0,1}\-\d{3,4}(\.\d{2})?$', text):
            return text
            
        # Clean all separator characters
        raw = re.sub(r'[\-\.]', '', text)
        
        # Correct common OCR confusion errors based on character positions
        raw = self.correct_confusion_characters(raw)
        
        # Standard Vietnamese plate match:
        # Group 1: 2-digit city code (e.g. 29, 30, 51)
        # Group 2: 1-2 letters, optionally followed by 1 digit (for motorcycles)
        # Group 3: 3 to 5 digits (e.g. 1234, 12345)
        match = re.match(r'^(\d{2})([A-Z]{1,2}\d{0,1})(\d{3,5})$', raw)
        if match:
            city_code = match.group(1)
            series = match.group(2)
            numbers = match.group(3)
            
            # Format numbers: if 5 digits, split as XXX.XX (e.g. 123.45)
            if len(numbers) == 5:
                numbers = f"{numbers[:3]}.{numbers[3:]}"
            return f"{city_code}{series}-{numbers}"
            
        return text

    def correct_confusion_characters(self, raw):
        """
        Heuristically corrects common OCR letter/digit confusions based on positions.
        """
        chars = list(raw)
        n = len(chars)
        if n < 5:
            return raw
            
        char_to_digit = {
            'O': '0', 'D': '0', 'Q': '0', 
            'I': '1', 'L': '1', 
            'S': '5', 'B': '8', 
            'G': '6', 'Z': '2', 'A': '4'
        }
        
        # 1. The first 2 characters must be digits (City Code)
        for i in range(2):
            if chars[i].isalpha() and chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]
                
        # 2. The last sequence of characters must be digits
        # Scan backwards and correct letters that are likely digits
        for i in range(n - 1, 2, -1):
            if chars[i].isalpha():
                if chars[i] in char_to_digit:
                    chars[i] = char_to_digit[chars[i]]
                else:
                    break # Stop if we hit a letter not resembling a digit
                    
        return "".join(chars)

    def read_plate(self, crop_image):
        """
        Performs OCR on a cropped license plate image.
        Sorts the detected text blocks to handle both 1-line and 2-line plates.
        
        Returns:
            text (str): The recognized and formatted plate string.
            confidence (float): Average confidence score of the detection.
        """
        if crop_image is None or crop_image.size == 0:
            return "", 0.0

        # Run OCR
        # We initialized use_textline_orientation/use_angle_cls at init, so we do not pass 'cls' here.
        # This avoids TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'.
        try:
            results = self.ocr.ocr(crop_image)
        except TypeError:
            # Fallback for older PaddleOCR versions that require cls parameter
            results = self.ocr.ocr(crop_image, cls=True)
        
        if not results or results[0] is None:
            return "", 0.0

        items = []
        
        # Support both new PaddleOCR v3.6+ dictionary format and older list format
        if isinstance(results[0], dict):
            res_dict = results[0]
            rec_texts = res_dict.get('rec_texts', [])
            rec_scores = res_dict.get('rec_scores', [])
            rec_polys = res_dict.get('rec_polys', [])
            
            for i in range(len(rec_texts)):
                text = rec_texts[i]
                conf = rec_scores[i]
                box = rec_polys[i] if i < len(rec_polys) else None
                if box is None or len(box) == 0:
                    continue
                
                # Calculate center coords
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                height = max(ys) - min(ys)
                
                items.append({
                    "cx": cx,
                    "cy": cy,
                    "height": height,
                    "text": text,
                    "confidence": conf
                })
        else:
            detections = results[0]
            for det in detections:
                box = det[0]
                text = det[1][0]
                conf = det[1][1]
                
                # Calculate center coords
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                height = max(ys) - min(ys)
                
                items.append({
                    "cx": cx,
                    "cy": cy,
                    "height": height,
                    "text": text,
                    "confidence": conf
                })

        # Group text items into rows (for 2-line plates)
        # Sort items by Y-coordinate first
        items.sort(key=lambda x: x["cy"])
        
        rows = []
        if items:
            current_row = [items[0]]
            avg_height = items[0]["height"]
            
            for item in items[1:]:
                # If Y-center difference is less than 60% of average height, group in same row
                if abs(item["cy"] - current_row[-1]["cy"]) < (avg_height * 0.6):
                    current_row.append(item)
                    # Update average height in row
                    avg_height = sum(x["height"] for x in current_row) / len(current_row)
                else:
                    rows.append(current_row)
                    current_row = [item]
                    avg_height = item["height"]
            rows.append(current_row)

        # Sort each row from left to right (by X-coordinate) and build text
        final_text_parts = []
        confidences = []
        
        for row in rows:
            row.sort(key=lambda x: x["cx"])
            row_text = "".join(x["text"] for x in row)
            final_text_parts.append(row_text)
            confidences.extend([x["confidence"] for x in row])

        # Join lines. If multiple lines, we can combine with a space or hyphen
        # but let's join them directly and let format_plate_text reformat it.
        combined_text = "".join(final_text_parts)
        formatted_text = self.format_plate_text(combined_text)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return formatted_text, avg_confidence
