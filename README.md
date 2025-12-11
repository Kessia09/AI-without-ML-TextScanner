# PyTesseract GUI Printed Text Scanner

This project is a GUI-based printed text scanner that uses PyTesseract for OCR.

## Features

* Load images from disk.
* Live camera input.
* Select Region of Interest (ROI) for OCR.
* Preprocess images (grayscale, median blur, adaptive thresholding).
* Overlay detected text on the image preview.
* Display extracted text in a side panel.

## Requirements

* Python 3.8+
* PyQt5
* OpenCV (opencv-python)
* Pillow
* pytesseract
* numpy

Install dependencies using pip:

```bash
pip install pyqt5 opencv-python pillow pytesseract numpy
```

## Setup (Windows)

Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract). Then configure the path in your code if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Usage

Run the Python script:

```bash
python <your_filename>.py
```

Steps:

1. Open the application.
2. Load an image or start the live camera.
3. (Optional) Enable 'Select ROI' and drag to select the region.
4. Click 'Run OCR' to extract text.
5. Extracted text appears in the right panel and overlayed on the image.
6. Use 'Clear OCR' to reset results.

## Notes

* The GUI is implemented with PyQt5.
* OCR preprocessing improves text recognition accuracy.
* The ROI overlay helps you focus OCR on specific parts of the image.
