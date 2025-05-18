#OCR CODE RUNNER/INTERPRETER 
#Uses Tesseract OCR to detect code text, write it in ocr.py, run ocr.py and display the output live
#By King N. Behimino 
#2022-05822
#For CMSC 191 Project 

#topics implemented:
# 1. Python + OpenCV image handling
# 2. Image manipulation and processing
# 3. Image segmentation and contour detection (preprocessing of webcam for segmentation, canny edge for contour detection)
# 4. Object tracking and motion analysis (bounding boxes)
# 5. OCR (tesseract)

#lab codes sources: 2.10, 2.11, 2.12
#sources used for code study and implementation: 
# 1. https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr 
# 2. https://docs.streamlit.io/develop/quick-reference/cheat-sheet
# 3. https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

#Own modifications/code: 
#1. Integrating streamlit 
#2. Opening and closing another python file for reading and writing code.
#3. multithreading for OCR 

#imports
import cv2
import pytesseract #MUST install pytesseract and tesseract OCR on local machine
import threading
import numpy as np
import subprocess
import streamlit as st #MUST install streamlit on local machine


# Initialize state variables
st.set_page_config(page_title="GUI")
st.title("OCR Written Code Interpreter")
st.subheader("Gets the text in the video via OCR and executes and displays the code.\n Made by King Behimino")

# Session state initialization
if "stopped" not in st.session_state:
    st.session_state.stopped = False
if "last_code" not in st.session_state:
    st.session_state.last_code = ""
if "last_output" not in st.session_state:
    st.session_state.last_output = ""

# Control buttons
col_btn1, col_btn2 = st.columns(2)
if col_btn1.button("Start"):
    st.session_state.stopped = False
if col_btn2.button("Stop"):
    st.session_state.stopped = True

cam1, cam2 = st.columns((3, 3))
video_1 = cam1.empty()
video_2 = cam2.empty()
text_col1, text_col2 = st.columns(2)
text_display = text_col1.empty()
execution_display = text_col2.empty()

last_processed_text = ""
processing_text = False
execution_result = ""

def process_ocr_text(text):
    #for processing ocr
    global processing_text, execution_result, last_processed_text
    
    # Avoid processing the same text multiple times
    if text.strip() == last_processed_text.strip():
        processing_text = False
        return
        
    last_processed_text = text

    try:
        # Write text to file
        with open("ocr.py", "w") as f:
            f.write(text)

        # Execute the code
        result = subprocess.run(["python", "ocr.py"],capture_output=True,text=True,timeout=3)
        execution_result = result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        execution_result = f"Error: {str(e)}"

    st.session_state.last_code = text
    st.session_state.last_output = execution_result
    processing_text = False

##-main--------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
frame_count = 0
ocr_interval = 15 #process every 15 frames
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while not st.session_state.stopped:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480)) #resizing frame for faster processing
    display_frame = frame.copy()

    if frame_count % ocr_interval == 0 and not processing_text:
        #image manipulation and processing
        grayscale = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
        gausBlur = cv2.GaussianBlur(grayscale, (5,5), 0)
        canny = cv2.Canny(gausBlur, 50,150)
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(canny, kernel, iterations=1)

        extractedText = pytesseract.image_to_string(dilation)
        # print("Extracted Text:\n")
        # print(extractedText)

        if extractedText.strip():
            # Get data for bounding boxes
            processing_text = True
            #separate thread processing for OCR
            thread = threading.Thread(target=process_ocr_text, args=(extractedText,))
            thread.daemon = True
            thread.start()

            data = pytesseract.image_to_data(dilation, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['level'])

            # Draw bounding boxes on the frame
            for i in range(n_boxes):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                conf = int(data['conf'][i])
                if conf > 40:  # confidence filter
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update Streamlit side panel for text and output
        text_display.code(last_processed_text or "No code yet", language="python")
        execution_display.code(execution_result or "No output yet", language="text")

    if execution_result:
    # Display up to 3 lines of the execution result
        lines = execution_result.strip().split('\n')[:3]
        for i, line in enumerate(lines):
            cv2.putText(
                display_frame, 
                line[:30],  # Limit length to fit on screen
                (10, 30 + i*30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
    frame_count += 1

    video_1.image(display_frame, channels="RGB", width=1200)
    video_2.image(dilation, channels="RGB", width=1200)