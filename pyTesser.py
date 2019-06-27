from PIL import Image
import pytesseract
from pytesseract import image_to_string
im = Image.open("asc.jpg")
text = pytesseract.image_to_string(im, lang="eng")
print(text)
