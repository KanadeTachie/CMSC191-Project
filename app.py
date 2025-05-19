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

cam1, cam2 = st.columns((1, 1))
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

def reconstruct_code_from_ocr(data):
    # Group text by approximate line position
    # this is for finding the indents in the code
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

    # Sort lines by vertical position and reconstruct code
    code_lines = []
    line_keys = sorted(lines.keys())
    
    # Calculate baseline indentation (leftmost text position)
    all_lefts = []
    for key in line_keys:
        line_words = lines[key]
        if line_words:
            # Get the leftmost position of each line
            leftmost = min(left for left, _ in line_words)
            all_lefts.append((key, leftmost))
    
    # Sort lines by their vertical position
    if all_lefts:
        # Find the most common left positions to identify indentation levels
        left_positions = [pos for _, pos in all_lefts]
        left_positions.sort()
        
        # Identify distinct indentation levels
        indent_levels = []
        for pos in left_positions:
            # Check if this position represents a new indent level
            is_new_level = True
            for known_pos in indent_levels:
                if abs(pos - known_pos) < 15:  # If within 15 pixels, consider same level
                    is_new_level = False
                    break
            if is_new_level:
                indent_levels.append(pos)
        
        indent_levels.sort()  # Sort indentation levels from left to right
    else:
        indent_levels = [0]
    
    for key in line_keys:
        # Sort words in the line by horizontal position
        line = sorted(lines[key])
        
        if not line:
            continue
            
        # Get left position of first word in line
        first_word_left = line[0][0]
        
        # Determine indentation level based on position
        indent_level = 0
        for i, level_pos in enumerate(indent_levels):
            if abs(first_word_left - level_pos) < 15:
                indent_level = i
                break
            elif first_word_left > level_pos:
                # If position is to the right of this level but not close enough
                # to match exactly, it might be the next level
                indent_level = i + 1
        
        # Cap indentation at 2 levels
        indent_level = min(indent_level, 2)
        line_text = " ".join(word for _, word in line)
        # Add appropriate indentation and append to code lines
        code_lines.append("    " * indent_level + line_text)
    
    # Join all lines with newlines
    return "\n".join(code_lines)

##-main--------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
frame_count = 0
ocr_interval = 5 #process every 15 frames
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

        data = pytesseract.image_to_data(dilation, output_type=pytesseract.Output.DICT)
        extractedText = reconstruct_code_from_ocr(data)
        
        # print("Extracted Text:\n")
        # print(extractedText)

        if extractedText.strip():
            # Get data for bounding boxes
            processing_text = True
            #separate thread processing for OCR
            thread = threading.Thread(target=process_ocr_text, args=(extractedText,))
            thread.daemon = True
            thread.start()

            n_boxes = len(data['level'])

            # Draw bounding boxes on the frame
            data = pytesseract.image_to_data(dilation, output_type=pytesseract.Output.DICT)
            for i in range(n_boxes):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                conf = int(data['conf'][i])
                if conf > 20:  # confidence filter
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update Streamlit GUI
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

    video_1.image(display_frame, channels="BGR", use_column_width=True)
    video_2.image(dilation, channels="RGB", use_column_width=True)