# PhotoPy
An OCR Python Code Runner/Interpreter. Inspired from Photomath mobile application, it uses Tesseract OCR to detect printed code and display the output live.

**Made for CMSC 191 Final Project**
## Developed by:
* **King Behimino** [@KanadeTachie](https://github.com/KanadeTachie)

## Topics implemented:
1. Python + OpenCV image handling
2. Image manipulation and processing
3. Image segmentation and contour detection (preprocessing of webcam for segmentation, canny edge for contour detection)
4. Object tracking and motion analysis (bounding boxes)
5. OCR (tesseract)

## Lab resources used:
1. **2.10** - Thresholding, Binarization & Adaptive Thresholding
2. **2.11** - Dilation, Erosion, Opening and Closing
3. **2.12** - Edge Detection & Image Gradients
4. **9.3** - Automatic Number-Plate Recognition


## Sources used for code study and implementation: 
1. https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr
2. https://docs.streamlit.io/develop/quick-reference/cheat-sheet
3. https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

## For Local Testing:
Fork this repo and type the following in the console
```bash
pip install pytesseract
pip install streamlit
```
Aside from the two, Tesseract OCR MUST be installed on the local machine.
Download the latest Tesseract OCR and install.
```bash
https://github.com/tesseract-ocr/tesseract
```
MUST add the installation folder to System PATH

 **To run the code locally, open a cmd towards the app.py path and type:**
 ```bash
streamlit run app.py
```
The images on the test images folder are to be saved on a separate device (phone, laptop, etc.) for live webcam testing.
Custom code text images can be made on another device, but it must not be handwritten.

## Current limitations of this program:
1. Has the difficulty of detecting handwritten code (as Tesseract OCR is not trained on handwritten code)
2. Some code with 5 lines or more will be difficult to read
3. Only reads code printed digitally from Paint
