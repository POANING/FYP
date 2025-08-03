import cv2
import tkinter as tk
from tkinter import Button, PhotoImage
from PIL import Image, ImageTk
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import time

root = tk.Tk()
root.geometry("1200x768")

face_cascade = cv2.CascadeClassifier('FYP\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('FYP\\haarcascade_eye.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

blink_thresh = 0.2
closed_thresh = 0.2  
blink_counter = 0
closed_counter = 0
prev_blink_state = False
prev_closed_state = False
prev_closed_time = None
concentration = 100.0

def start_detection():
    global is_running, cap, blink_counter, closed_counter, prev_blink_state, prev_closed_state, prev_closed_time, concentration
    is_running = True
    blink_counter = 0
    closed_counter = 0
    prev_blink_state = False
    prev_closed_state = False
    prev_closed_time = None
    concentration = 100.0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open the camera.")
        return

    root.after(10, display_frames)

def calculate_EAR(eye):
    if len(eye) != 6:
        return None
    
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    x = dist.euclidean(eye[0], eye[3])

    ear = (y1 + y2) / (2 * x)
    return ear

def detect_blink(eye_landmarks):
    left_eye = eye_landmarks[0:6]
    right_eye = eye_landmarks[6:12]

    left_ear = calculate_EAR(left_eye)
    right_ear = calculate_EAR(right_eye)

    avg_ear = (left_ear + right_ear) / 2

    if avg_ear < blink_thresh:
        return True
    else:
        return False
    
def update_concentration_value():
    global concentration
    concentration_text = "{}%".format(int(concentration * 100))
    concentration_var.set(concentration_text)

    if concentration == 0.0:
        concentration_indicator.config(bg="red")
    elif concentration <= 0.5:
        concentration_indicator.config(bg="orange")
    elif concentration <= 0.99:
        concentration_indicator.config(bg="green")
    else:
        concentration_indicator.config(bg="dark green")

concentration_drop_count = 0

concentration_fall = False

last_blink_time = time.time()

closed_counter = 0
blink_counter = 0
concentration = 0.0
last_blink_time = 0.0
drop_count = 0
prev_concentration = 0.0
drop_detected = False

concentration_drop_count_var = tk.StringVar()
concentration_drop_count_var.set(str(concentration_drop_count))

drop_count_sentence = tk.Label(root, text="You have {} times \n not concentrate during class".format(concentration_drop_count_var.get()), width=30, height=5, bd=1, relief="solid")
drop_count_sentence.config(font=("Courier", 12))
drop_count_sentence.place(x=850, y=550)

def update_concentration_drop_count():
    global concentration_drop_count
    concentration_drop_count += 1
    
    drop_count_sentence.config(text="You have {} times \n not concentrate during class".format(concentration_drop_count_var.get()))
    
def update_drop_count_label():
    global concentration_drop_count
    concentration_drop_count_var.set(str(concentration_drop_count))
    drop_count_sentence.config(text="You have {} times \n not concentrate during class".format(concentration_drop_count_var.get()))

def analyze_concentration():
    global closed_counter, blink_counter, concentration, last_blink_time, rise_count, drop_count, prev_concentration, drop_detected

    drop_detected = False  

    if closed_counter > 0:
        concentration = (blink_counter / closed_counter) * (1 - blink_counter / 100)
        if concentration < 0:
            concentration = 0
        elif concentration > 1:
            concentration = 1
    else:
        concentration = 1.0
    
    current_time = time.time()
    time_since_last_blink = current_time - last_blink_time

    if time_since_last_blink > 3:
        increase_percentage = int(time_since_last_blink / 3)  # Calculate the number of 3-second intervals
        concentration += (0.01 * increase_percentage)
        
    if concentration > 1:
        concentration = 1  
    elif concentration < 0:
        concentration = 0  

    if closed_counter > 0 and current_time - prev_closed_time > 10:
        concentration = 0
    
    if concentration < 0.5 and prev_concentration >= 0.5:
        drop_count += 1
        drop_detected = True
    
    if drop_detected:
        update_concentration_drop_count() 
        update_drop_count_label() 

    update_concentration_value()

    if blink_counter == 0:
        last_blink_time = current_time  

    prev_concentration = concentration


def detect_blinks(frame, faces):
    global blink_counter, prev_blink_state, closed_counter, prev_closed_state, prev_closed_time

    closed_eye_counter = 0  

    for (x, y, w, h) in faces:

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = calculate_EAR(shape[36:42])
        right_eye = calculate_EAR(shape[42:48])

        avg_ear = (left_eye + right_eye) / 2
        if avg_ear < blink_thresh:
            if not prev_blink_state:
                blink_counter += 1
                prev_blink_state = True
                if prev_closed_state and prev_closed_time is not None:
                    elapsed_time = time.time() - prev_closed_time
                    if elapsed_time > 30:
                        closed_counter -= 1
        else:
            prev_blink_state = False
            
        if detect_closed(left_eye, right_eye):
            if not prev_closed_state:
                prev_closed_state = True
                prev_closed_time = time.time()
                closed_eye_counter += 1
        else:
            prev_closed_state = False

    closed_counter += closed_eye_counter
    
    analyze_concentration()

def detect_closed(left_eye, right_eye):
    if left_eye is None or right_eye is None:
        return False

    closed_thresh = 0.2

    if left_eye < closed_thresh and right_eye < closed_thresh:
        return True

    return False

def display_frames():
    global is_running, cap, blink_counter, concentration

    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)

        image_tk = ImageTk.PhotoImage(image)

        label.config(image = image_tk)
        label.image = image_tk

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        detect_blinks(frame, faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame_rgb, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

            concentration_text = "{}%".format(int(concentration * 100))
            text_width, text_height = cv2.getTextSize(concentration_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(frame_rgb, concentration_text, (x + int((w - text_width) / 2), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        image_rgb = Image.fromarray(frame_rgb)
        image_rgb = ImageTk.PhotoImage(image_rgb)

        label.config(image=image_rgb)
        label.image = image_rgb

    if is_running:
        root.after(10, display_frames)
    else:
        cap.release()
        cv2.destroyAllWindows()

def stop_detection():
    global is_running
    is_running = False

image_path_1= "FYP\\Picture1.jpg"
image_path_2= "FYP\\Picture2.jpg"

image_1 = Image.open(image_path_1)
image_2 = Image.open(image_path_2)

image_1 = image_1.resize((100, 100))

image_tk_1 = ImageTk.PhotoImage(image_1)
image_tk_2 = ImageTk.PhotoImage(image_2)

label_1 = tk.Label(root, image=image_tk_1)
label_1.place(x=900, y=10)

label_2 = tk.Label(root, image=image_tk_2)
label_2.place(x=20, y=10)

main_frame = tk.Label(root, text="REAL TIME CONCENTRATION ANALYZER FOR ONLINE CLASS", fg="black")
main_frame.config(font=("Helvetica", 12))
main_frame.place(x=150, y=30)

label = tk.Label(root, bg="black", width=66)
label.place(x=150, y=55)

main_frame = tk.Label(root, text=" We track and monitor your activities during class ", fg="black")
main_frame.config(font=("Helvetica", 12))
main_frame.place(x=180, y=76)

main_frame = tk.Label(root, text="REAL TIME STATUS STUDENT CONCENTRATION ", fg="black")
main_frame.config(font=("Helvetica", 12))
main_frame.place(x=800, y=120)

concentration_var = tk.StringVar()
concentration_var.set("100%")

concentration_label = tk.Label(root, textvariable=concentration_var)
concentration_label.place(x=350, y=300)

label = tk.Label(root)
label.place(x=150, y=150, width=600, height=480)

concentration_label = tk.Label(root, text="LEVEL OF CONCENTRATION", fg="black")
concentration_label.config(font=("Courier", 10))
concentration_label.place(x=850, y=160)

concentration_indicator = tk.Label(root, bg="gray", relief=tk.SUNKEN)
concentration_indicator.place(x=850, y=180, width=80, height=40)

concentration_label = tk.Label(root, text="PERCENTAGE", fg="black")
concentration_label.config(font=("Courier", 10))
concentration_label.place(x=1050, y=160)

concentration_value = tk.Label(root, textvariable=concentration_var, fg="black", bg="white", width=5, height=1, bd=1, relief="solid")
concentration_value.config(font=("Courier", 20))
concentration_value.place(x=1050, y=180)

concentration_label = tk.Label(root, text="Indicator: ", fg="black")
concentration_label.config(font=("Courier", 12))
concentration_label.place(x=850, y=240)

concentration_label1 = tk.Label(root, bg="red", width=5, height=3, bd=1, relief="solid")
concentration_label1.place(x=850, y=270)

concentration_label2 = tk.Label(root, bg="orange", width=5, height=3,bd=1, relief="solid")
concentration_label2.place(x=850, y=325)

concentration_label3 = tk.Label(root, bg="green", width=5, height=3, bd=1, relief="solid")
concentration_label3.place(x=850, y=380)

concentration_label4 = tk.Label(root, bg="#006400", width=5, height=3, bd=1, relief="solid")
concentration_label4.place(x=850, y=435)

concentration_text1 = tk.Label(root, text="No concentration", fg="black")
concentration_text1.config(font=("Courier", 12))
concentration_text1.place(x=900, y=280)

concentration_text2 = tk.Label(root, text="Less concentrate", fg="black")
concentration_text2.config(font=("Courier", 12))
concentration_text2.place(x=900, y=340)

concentration_text3 = tk.Label(root, text="Half concentrate", fg="black")
concentration_text3.config(font=("Courier", 12))
concentration_text3.place(x=900, y=390)

concentration_text4 = tk.Label(root, text="Concentrate", fg="black")
concentration_text4.config(font=("Courier", 12))
concentration_text4.place(x=900, y=445)

concentration_label = tk.Label(root, text="ANALYSIS: ", fg="black")
concentration_label.config(font=("Courier", 10))
concentration_label.place(x=850, y=525)

start_button = tk.Button(root, text="Start Analyze", fg="black", width=15, height=1, bd=1, relief="solid", command=start_detection)
start_button.config(font=("Courier", 15))
start_button.place(x=250, y=650)

stop_button = tk.Button(root, text="Stop Analyze ", fg="black", width=15, height=1, bd=1, relief="solid", command=stop_detection)
stop_button.config(font=("Courier", 15))
stop_button.place(x=460, y=650)

root.mainloop()