import cv2
import os
import pandas as pd
import tkinter as tk
from tkinter import Label, Button
import csv
from PIL import Image, ImageTk

# Load the Excel file
df = pd.read_excel('TinyCategories.xlsx', engine='openpyxl', header=None)

# Extract the first column and convert it to a list of strings
groups = df.iloc[:, 0].astype(str).tolist()
idx = groups.index("haircut_scissor") + 1
groups = groups[idx:]

# Print the list
print(groups)

# Video folder setup
video_dir = "video"
group_index = 0  # Start at first group
video_index = 0  # Start at first video

# Load first video
cap = cv2.VideoCapture(os.path.join(video_dir, groups[group_index], os.listdir(os.path.join(video_dir, groups[group_index]))[video_index]))

# Create GUI window
root = tk.Tk()
root.title("Video Annotation")

# Create a blank image to initialize the label
blank_img = ImageTk.PhotoImage(Image.new("RGB", (640, 360), (0, 0, 0)))
video_label = Label(root, image=blank_img)
video_label.image = blank_img  # Prevent garbage collection
video_label.grid(row=0, column=0, columnspan=7)

# Global variable to hold the image reference
img = None
frame_updating = False  # Flag to prevent multiple updates

# Create labels for annotation options
race_var = tk.StringVar(value="White")
sex_var = tk.StringVar(value="Male")
age_var = tk.StringVar(value="Adult")
multiple_people_var = tk.StringVar(value="No")
face_visible_var = tk.StringVar(value="Yes")

# Annotation buttons
def set_race(race):
    race_var.set(race)

def set_sex(sex):
    sex_var.set(sex)

def set_age(age):
    age_var.set(age)

def set_multiple_people(choice):
    multiple_people_var.set(choice)

def set_face_visible(choice):
    face_visible_var.set(choice)

# Create Next button
def save_and_next():
    global video_index, group_index, cap  # Declare variables as global

    # Save annotations to CSV
    with open('annotations.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([groups[group_index], os.listdir(os.path.join(video_dir, groups[group_index]))[video_index], race_var.get(), sex_var.get(), age_var.get(), multiple_people_var.get(), face_visible_var.get()])
    
    # Move to next video or group
    video_index += 1
    if video_index >= len(os.listdir(os.path.join(video_dir, groups[group_index]))):
        video_index = 0
        group_index += 1
        if group_index >= len(groups):
            group_index = 0  # Loop back to first group
    # Load next video
    cap.release()  # Release the previous video capture
    cap = cv2.VideoCapture(os.path.join(video_dir, groups[group_index], os.listdir(os.path.join(video_dir, groups[group_index]))[video_index]))
    
    # Update the frame once
    update_frame()

# Skip button function
def skip_video():
    global video_index, group_index, cap  # Declare variables as global

    # Move to next video or group without saving annotations
    video_index += 1
    if video_index >= len(os.listdir(os.path.join(video_dir, groups[group_index]))):
        video_index = 0
        group_index += 1
        if group_index >= len(groups):
            group_index = 0  # Loop back to first group
    # Load next video
    cap.release()  # Release the previous video capture
    cap = cv2.VideoCapture(os.path.join(video_dir, groups[group_index], os.listdir(os.path.join(video_dir, groups[group_index]))[video_index]))
    
    # Update the frame once
    update_frame()

def skip_cat():
    global video_index, group_index, cap  # Declare variables as global
    video_index = 0
    group_index += 1

    # Load next video
    cap.release()  # Release the previous video capture
    cap = cv2.VideoCapture(os.path.join(video_dir, groups[group_index], os.listdir(os.path.join(video_dir, groups[group_index]))[video_index]))
    
    # Update the frame once
    update_frame()

# Create annotation buttons
race_buttons = ["White", "Black", "Asian", "Indian", "ME", "Hispanic", "Unknown"]
sex_buttons = ["Male", "Female", "Unknown"]
age_buttons = ["Child", "Teen", "Adult", "MA", "Senior", "Unknown"]
people_buttons = ["One Person", "More than One Person"]
face_buttons = ["Face visible", "Face not Visible"]

# Function to create buttons with padding
def create_buttons(button_list, command_func, row_num):
    for col, button in enumerate(button_list):
        btn = Button(root, text=button, command=lambda b=button: command_func(b))
        btn.grid(row=row_num, column=col, padx=5, pady=14)  # Use padx and pady for margin

# Create annotation buttons with margin
create_buttons(race_buttons, set_race, 1)
create_buttons(sex_buttons, set_sex, 2)
create_buttons(age_buttons, set_age, 3)
create_buttons(people_buttons, set_multiple_people, 4)
create_buttons(face_buttons, set_face_visible, 5)

# Next button
next_button = Button(root, text="Next", command=save_and_next)
next_button.grid(row=6, column=len(age_buttons) - 1, pady=14)  # Apply padding for margin

# Skip button
skip_button = Button(root, text="Skip Vid", command=skip_video)
skip_button.grid(row=6, column=len(age_buttons) - 2, pady=14)  # Position the skip button next to the "Next" button

# Skip category button
skip_cat_button = Button(root, text="Skip Cat", command=skip_cat)
skip_cat_button.grid(row=6, column=len(age_buttons) - 3, pady=14)  # Position the skip button next to the "Next" button

# Function to update video frames
def update_frame():
    global img, frame_updating
    if frame_updating:  # Check if a frame update is already in progress
        return

    frame_updating = True  # Set flag to prevent multiple updates

    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 360))
        img = ImageTk.PhotoImage(Image.fromarray(frame))  # Update image reference

        # Update video label
        video_label.config(image=img)
        video_label.image = img  # Keep reference to avoid garbage collection

        # Re-enable frame update
        frame_updating = False

        # Schedule the next frame update
        root.after(100, update_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
        frame_updating = False  # Reset the flag for the next video
        # root.after(100, update_frame)

# Call update_frame() after the window has loaded
root.after(500, update_frame)

# Run the GUI loop
root.mainloop()
