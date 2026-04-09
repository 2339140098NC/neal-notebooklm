import fitz # PyMuPDF

doc = fitz.open("A History of Japan.pdf")

for page in doc:
    print(page.get_text())