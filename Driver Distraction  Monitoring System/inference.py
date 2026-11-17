import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from resnet_50 import ResNet50
import sys

video_path = None
running = False
cap = None

# intialized ResNet-50 for set trained weights to the model
model = ResNet50(input_shape = (128,128, 3), classes = 10)

# Load the weights
model.load_weights('restnet_50.weights.h5')
print("Model weights loaded successfully!")

# Function to open a file dialog and select a video file
def choose_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_path:
        file_label.config(text=f"Selected: {video_path}")

# Function to display frames in Tkinter window
def show_frame():
    global running, cap
    if not running or cap is None:
        return
    
    ret, frame = cap.read()
    if not ret:
        stop_monitoring()
        return

    # Resize frame for display
    display_width, display_height = 800, 450
    frame = cv2.resize(frame, (display_width, display_height))

    data = cv2.resize(frame,(128,128))
    data = data.astype("float") / 255.0
    data = np.expand_dims(data, axis=0)

    # Predict using the model
    prediction = model.predict(data)
    # Names for classification
    Name = ["safe driving","texting - right","talking on the phone - right","texting - left","talking on the phone - left","operating the radio","drinking","reaching behind","hair and makeup","talking to passenger"]
    predicted_class = Name[prediction.argmax()]
    print("predicted_class",predicted_class)

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # Red color for text
    text = f"Predicted class: {predicted_class}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    cv2.putText(frame, f'Driver Distraction Monitoring: {predicted_class}', (30,30),
            cv2.FONT_HERSHEY_SIMPLEX,1, (0,0, 255), 2)


    # Convert frame to ImageTk format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Call show_frame again after 10ms
    root.after(10, show_frame)

# Function to start video streaming
def start_monitoring():
    global running, cap
    if video_path:
        running = True
        cap = cv2.VideoCapture(video_path)
        show_frame()
    else:
        file_label.config(text="Please select a video file first!", fg="red")

# Function to stop video streaming
def stop_monitoring():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    video_label.config(image="")
    sys.exit(1)

# Initialize Tkinter window
root = Tk()
root.title("Driver Distraction Monitoring System")
root.geometry("1024x600")
root.resizable(True, True)

# Style configuration
style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
style.configure("Treeview", font=("Arial", 10))

# Create Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Tab 1: Video Monitoring
monitor_tab = Frame(notebook)
notebook.add(monitor_tab, text="Monitoring")


# Video selection frame
file_frame = Frame(monitor_tab)
file_frame.pack(fill="x", pady=5)

file_label = Label(file_frame, text="No file selected", font=("Arial", 12))
file_label.pack(side="left", padx=10)
file_button = Button(file_frame, text="Choose File", command=choose_file, font=("Arial", 12), bg="blue", fg="white")
file_button.pack(side="right", padx=10)

# Video display frame
video_frame = Frame(monitor_tab, bg="black")
video_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Video frame title label
title_label = Label(video_frame, text="Driver Distraction Monitoring System", font=("Arial", 18, "bold"), bg="black", fg="white")
title_label.pack(fill="x", pady=10)  # Add some padding to create space above the video

video_label = Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# Start and Stop buttons
button_frame = Frame(monitor_tab)
button_frame.pack(pady=20)

start_button = Button(button_frame, text="Start", command=start_monitoring, font=("Arial", 14), bg="green", fg="white")
start_button.pack(side="left", padx=10)

stop_button = Button(button_frame, text="Stop", command=stop_monitoring, font=("Arial", 14), bg="red", fg="white")
stop_button.pack(side="left", padx=10)

root.mainloop()

