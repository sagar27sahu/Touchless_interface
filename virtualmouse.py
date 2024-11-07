import pyautogui

# Get screen dimensions for scaling cursor movements
screen_width, screen_height = pyautogui.size()


def VirtualMouse(img, imgRGB, results, Draw, mphands, hands):
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw landmarks on the image
        Draw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)

        # Get landmarks of the right hand
        landmarks = hand_landmarks.landmark
        index_finger_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        # Get the dimensions of the image
        img_height, img_width, _ = img.shape

        # Convert normalized landmark coordinates to pixel coordinates
        x_pixel = int(index_finger_tip.x * img_width)
        y_pixel = int(index_finger_tip.y * img_height)

        # Map pixel coordinates to screen coordinates
        screen_x = int(screen_width * (index_finger_tip.x))
        screen_y = int(screen_height * (index_finger_tip.y))

        # Move the mouse cursor to the mapped screen position
        pyautogui.moveTo(screen_x, screen_y)

        # Display a circle at the index finger tip position on the image
        cv2.circle(img, (x_pixel, y_pixel), 10, (255, 0, 0), cv2.FILLED)

    # Optionally display a message indicating the virtual mouse is active
    cv2.putText(img, "Virtual Mouse Active", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)