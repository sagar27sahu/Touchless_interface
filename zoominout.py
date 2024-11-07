import math
import cv2
def ZoomInOut(img, imgRGB, results, Draw, mphands, hands):
    # Ensure that two hands are detected before zooming functionality
    if len(results.multi_hand_landmarks) == 2:
        # Extract the landmark points for both hands
        hand_landmarks = results.multi_hand_landmarks

        # Get the coordinates of the tips of the index fingers of both hands (landmark 8)
        left_hand_landmarks = hand_landmarks[0].landmark
        right_hand_landmarks = hand_landmarks[1].landmark

        # Calculate the distance between the tips of the index fingers of both hands
        x1, y1 = left_hand_landmarks[8].x * img.shape[1], left_hand_landmarks[8].y * img.shape[0]
        x2, y2 = right_hand_landmarks[8].x * img.shape[1], right_hand_landmarks[8].y * img.shape[0]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Define zoom limits and sensitivity
        zoom_min_distance = 50  # Minimum distance to consider zoom in
        zoom_max_distance = 300  # Maximum distance to consider zoom out
        zoom_sensitivity = 2  # Adjust this value to change zoom sensitivity

        # Determine zoom factor based on distance
        zoom_factor = (distance - zoom_min_distance) / (zoom_max_distance - zoom_min_distance)

        # Clamp zoom factor to a sensible range (0.5 to 2 for example)
        zoom_factor = max(0.5, min(2.0, zoom_factor))

        # Display the zoom factor on the screen for debugging
        cv2.putText(img, f'Zoom Factor: {zoom_factor:.2f}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Here you can apply the zoom to the image or to a target window as per your application.
        # Example: You can use OpenCV's resize function to simulate zoom.
        # Note: Adjust this logic based on your needs.

        # Get the center of the frame for zooming in and out
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        # Resize the image based on the zoom factor (for demonstration purposes)
        zoomed_img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        # Get dimensions of the zoomed image
        zoomed_height, zoomed_width = zoomed_img.shape[:2]

        # Calculate the region to display on the screen to maintain the center focus
        start_x = max(0, (zoomed_width // 2) - center_x)
        start_y = max(0, (zoomed_height // 2) - center_y)

        end_x = start_x + img.shape[1]
        end_y = start_y + img.shape[0]

        # Crop the zoomed image to fit the screen size
        cropped_zoomed_img = zoomed_img[start_y:end_y, start_x:end_x]

        # Display the zoomed and cropped image
        cv2.imshow('Zoomed Control', cropped_zoomed_img)
    else:
        # If less than two hands, display the original image
        cv2.imshow('Zoomed Control', img)