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
# Integrating streamlit GUI
# Custom image manipulation and processing settings
# Opening and closing another python file for reading and writing code.
# multithreading for OCR
# OCR functionality

#imports
import cv2
import pytesseract #MUST install pytesseract and tesseract OCR on local machine
import threading
import numpy as np
import subprocess
import streamlit as st #MUST install streamlit on local machine


# Initialize state variables -----------------------------------------------
st.set_page_config(page_title="GUI")
st.title("OCR Written Code Interpreter")
st.subheader("Gets the text in the video via OCR and executes and displays the code.\n Made by King Behimino")

# Session state initialization ----------------------------------------------
if "stopped" not in st.session_state:
    st.session_state.stopped = False
if "paused" not in st.session_state: 
    st.session_state.paused = False
if "last_code" not in st.session_state:
    st.session_state.last_code = ""
if "last_output" not in st.session_state:
    st.session_state.last_output = ""

# Streamlit Buttons ---------------------------------------------------------
col_btn1, col_btn2, col_btn3 = st.columns(3)
if col_btn1.button("Start"):
    st.session_state.stopped = False
    st.session_state.paused = False 
if col_btn2.button("Pause/Resume"):
    st.session_state.paused = not st.session_state.paused
if col_btn3.button("Stop"):
    st.session_state.stopped = True

cam1, cam2 = st.columns((1, 1))
video_1 = cam1.empty()
video_2 = cam2.empty()
text_col1, text_col2 = st.columns(2)
text_display = text_col1.empty()
execution_display = text_col2.empty()

last_processed_text = ""
processing_text = False
execution_result = ""

# For processing ocr----------------------------------------------------------------------------------------
def process_ocr_text(text):
    
    global processing_text, execution_result, last_processed_text
    
    if text.strip() == last_processed_text.strip():                                             # Avoid processing the same text multiple times
        processing_text = False
        return
    last_processed_text = text

    try:
        with open("ocr.py", "w") as f:
            f.write(text)

        result = subprocess.run(["python", "ocr.py"],capture_output=True,text=True,timeout=3)                                   # Execute code
        execution_result = result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        execution_result = f"Error: {str(e)}"

    st.session_state.last_code = text
    st.session_state.last_output = execution_result
    processing_text = False

# For finding the indents in the code-----------------------------------------------------------------------------
def reconstruct_code_from_ocr(data):
    lines = {}
    for i in range(len(data['level'])):
        conf = int(data['conf'][i])
        if conf > 40: 
            top = data['top'][i]
            word = data['text'][i].strip()
            left = data['left'][i]
            if word:
                found_line = False
                for key in lines:
                    if abs(key - top) <= 10: 
                        lines[key].append((left, word))
                        found_line = True
                        break
                if not found_line:
                    lines[top] = [(left, word)]

    
    code_lines = []                                                                        # Sort lines by vertical position and reconstruct code
    line_keys = sorted(lines.keys()) 
    all_lefts = []                                                                         # Calculate baseline indentation (leftmost text position)
    for key in line_keys:
        line_words = lines[key]
        if line_words:
            leftmost = min(left for left, _ in line_words)                                 # Get the leftmost position of each line
            all_lefts.append((key, leftmost))
    
    
    if all_lefts:                                                                          # Sort lines by their vertical position
        left_positions = [pos for _, pos in all_lefts]                                     # Find the most common left positions to identify indentation levels
        left_positions.sort()
        
        
        indent_levels = []                                                                 # Identify distinct indentation levels
        for pos in left_positions:
            is_new_level = True                                                            # Check if this position represents a new indent level
            for known_pos in indent_levels:
                if abs(pos - known_pos) < 15:                                              # If within 15 pixels, consider same level
                    is_new_level = False
                    break
            if is_new_level:
                indent_levels.append(pos)
        
        indent_levels.sort()                                                               # Sort indentation levels from left to right
    else:
        indent_levels = [0]
    
    for key in line_keys:
        line = sorted(lines[key])                                                          # Sort words in the line by horizontal position
        if not line:
            continue
        first_word_left = line[0][0]                                                       # Get left position of first word in line
        
        indent_level = 0                                                                   # Determine indentation level based on position
        for i, level_pos in enumerate(indent_levels):
            if abs(first_word_left - level_pos) < 15:
                indent_level = i
                break
            elif first_word_left > level_pos:                                               
                indent_level = i + 1

        indent_level = min(indent_level, 2)                                               # Cap indentation at 2 levels
        line_text = " ".join(word for _, word in line)
        code_lines.append("    " * indent_level + line_text)                              # Add appropriate indentation and append to code lines
    
    return "\n".join(code_lines)

##-main--------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
frame_count = 0
ocr_interval = 10                                                                         # process every 10 frames. Change to increase or decrease speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))
display_frame = frame.copy()
dilation = None 

while not st.session_state.stopped:

    if not st.session_state.paused:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        display_frame = frame.copy()
    
    if frame_count % ocr_interval == 0 and not processing_text:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                         #image manipulation and processing part
        gausBlur = cv2.GaussianBlur(grayscale, (5,5), 0)
        canny = cv2.Canny(gausBlur, 50,150)
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(canny, kernel, iterations=1)

        data = pytesseract.image_to_data(dilation, output_type=pytesseract.Output.DICT)
        extractedText = reconstruct_code_from_ocr(data)

        if extractedText.strip():                                                                  # Get data for bounding boxes
            processing_text = True
            thread = threading.Thread(target=process_ocr_text, args=(extractedText,))
            thread.daemon = True
            thread.start()

            n_boxes = len(data['level'])  

            data = pytesseract.image_to_data(dilation, output_type=pytesseract.Output.DICT)         # Draw bounding boxes on the frame
            for i in range(n_boxes):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                conf = int(data['conf'][i])
                if conf > 20:                                                                       # confidence filter
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_display.code(last_processed_text or "No code yet", language="python")
        execution_display.code(execution_result or "No output yet", language="text")

    if execution_result:
        lines = execution_result.strip().split('\n')[:3]                                            # Display up to 3 lines of the execution result
        for i, line in enumerate(lines):
            cv2.putText(
                display_frame, 
                line[:30],                                                                          # Limit length to fit on screen
                (10, 30 + i*30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
    frame_count += 1

    video_1.image(display_frame, channels="BGR", use_container_width=True)
    if dilation is not None:
        video_2.image(dilation, channels="RGB", use_container_width=True)