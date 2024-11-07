import cv2
from cvzone.HandTrackingModule import HandDetector
from time import time
from pynput.keyboard import Controller
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width

keyboard = Controller()

# Initialize HandDetector with confidence level
detector = HandDetector(detectionCon=0.8)

finalText = ""

# Define keys for the virtual keyboard (same size for all keys)
keys = [["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "<=="],
        ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
        ["z", "x", "c", "v", "b", "n", "m", " "],
        ["ESC"]]  # Added the ESC button in a new row

# Set uniform key size for all keys
button_size = [85, 85]
buttonList = []


# Function to add a blue-grayish gradient background to the interface
def add_gradient_background(img):
    overlay = img.copy()
    for i in range(img.shape[0]):
        # Blue-gray gradient, top starts more gray and fades into blue as it goes down
        color = (200 - int(i / img.shape[0] * 50), 200 - int(i / img.shape[0] * 50), 255 - int(i / img.shape[0] * 100))
        cv2.line(overlay, (0, i), (img.shape[1], i), color, 1)
    return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)


# Function to draw all buttons with improved design
def draw_all_buttons(img, buttonList):
    for button in buttonList:
        x, y = button.first_pos
        w, h = button.btn_size

        # Add a drop shadow
        shadow_offset = 5
        cv2.rectangle(img, (x + shadow_offset, y + shadow_offset), (x + w + shadow_offset, y + h + shadow_offset),
                      (150, 150, 150), cv2.FILLED)  # Light gray shadow

        # Draw rounded rectangle buttons
        cv2.rectangle(img, button.first_pos, (x + w, y + h), (240, 240, 240), cv2.FILLED)  # Light gray button
        cv2.rectangle(img, button.first_pos, (x + w, y + h), (180, 180, 180), 3, cv2.LINE_AA)  # Border

        # Button text with shadow for better contrast
        cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 6)  # Black shadow
        cv2.putText(img, button.text, (x + 18, y + 58), cv2.FONT_HERSHEY_PLAIN, 4, (50, 50, 255), 3)  # Blue text


# Button class to define the properties of each button
class Button:
    def __init__(self, first_pos, text):
        self.first_pos = first_pos
        self.text = text
        self.btn_size = button_size  # All buttons now have uniform size


# Create buttons for each key and append to the buttonList
for i in range(len(keys)):
    for x, key in enumerate(keys[i]):  # Enumerate to get position of each key
        buttonList.append(Button([100 * x + 80, 100 * i + 10], key))

# Initialize a dictionary to track time spent on each key
press_time = {}

# Calculate total height dynamically based on the number of rows and add some extra space
total_rows = len(keys)
window_height = (total_rows * 100) + 100  # Adding 100 pixels for hand detection space

# Main loop for processing video frames
while True:
    success, img = cap.read()  # Capture frame from webcam
    img = cv2.flip(img, 1)  # Flip image horizontally for correct hand display

    # Crop the window height to match only the keys area with extra space for hand detection
    img = img[0:window_height, 0:1280]  # Increased height for better hand detection

    # Add blue-grayish gradient background
    img = add_gradient_background(img)

    hands, img = detector.findHands(img)  # Detect hands and landmarks

    draw_all_buttons(img, buttonList)  # Draw virtual keyboard with enhanced design

    # Check if hands are detected
    if hands:
        lmList = hands[0]['lmList']  # Get landmark list for the first hand

        # Check for the index finger tip position (landmark 8) on the virtual keyboard
        for button in buttonList:
            x, y = button.first_pos
            w, h = button.btn_size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                # Darken the button to indicate hovering
                cv2.rectangle(img, button.first_pos, (x + w, y + h), (150, 150, 150), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 3)

                # Track time spent hovering over the button
                if button.text not in press_time:
                    press_time[button.text] = time()  # Record the start time

                # Check if the index finger tip has stayed for 1 second
                if time() - press_time[button.text] > 1:
                    if button.text == "ESC":  # Terminate the program if ESC is pressed
                        cap.release()  # Release the camera
                        cv2.destroyAllWindows()  # Close all OpenCV windows
                        exit(0)  # Exit the program
                    elif button.text == "<==":  # Check if it's the backspace key
                        if finalText:  # Only remove if finalText is not empty
                            finalText = finalText[:-1]  # Remove the last character from the virtual display
                            keyboard.press('\b')  # Simulate backspace key press in external applications
                            keyboard.release('\b')  # Release the backspace key
                    else:
                        finalText += button.text  # Append the character to finalText
                        keyboard.press(button.text)  # Simulate typing on the keyboard
                        keyboard.release(button.text)  # Release the key after pressing
                    cv2.rectangle(img, button.first_pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)  # Change to green
                    cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 3)
                    press_time.pop(button.text)  # Reset the timer after pressing the key

            else:
                # Reset the timer if the finger moves off the button
                if button.text in press_time:
                    press_time.pop(button.text)

    # Display the typed text on the screen
    cv2.putText(img, finalText, (165, 300), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)

    # Show the video feed with virtual keyboard
    cv2.imshow("Blue-Gray Virtual Keyboard", img)
    cv2.waitKey(1)