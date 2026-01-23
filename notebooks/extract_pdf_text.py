from pdfminer.high_level import extract_text
import os

pdf_path = "data/Détection de Fraude.pdf"
text = extract_text(pdf_path)

with open("data/pdf_content.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Text extracted and saved to data/pdf_content.txt")
