import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmlist, bbox

    def fingersUp(self,lmList):
        fingers = []
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    PTime = 0
    CTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        CTime = time.time()
        fps = 1 / (CTime - PTime)
        PTime = CTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()



# import cv2
# import mediapipe as mp
#
# class HandDetector():
#     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon
#
#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
#         self.mpDraw = mp.solutions.drawing_utils
#
#     def findHands(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)
#         # print(results.multi_hand_landmarks)
#
#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
#         return img
#
#     def findPosition(self, img, handNo=0, draw=True):
#
#         lmlist = []
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]
#             for id, lm in enumerate(myHand.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmlist.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
#         return lmlist
#
#     def fingersUp(self, lmlist):
#         fingers = []
#         # Thumb
#         if lmlist[4][1] > lmlist[3][1]:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#         # Fingers
#         for id in range(1, 5):
#             if lmlist[id*4][2] < lmlist[id*4 - 2][2]:
#                 fingers.append(1)
#             else:
#                 fingers.append(0)
#         return fingers
#
#
#
# def main():
#     detector = HandDetector()
#     cap = cv2.VideoCapture(0)
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         img = detector.findHands(img)
#         lmlist = detector.findPosition(img)
#         if len(lmlist) != 0:
#             fingers = detector.fingersUp(lmlist)
#             print(fingers)
#         cv2.imshow("Image", img)
#         if cv2.waitKey(1)  == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()


# import cv2
# import mediapipe as mp
#
#
# class HandTrackingModule:
#     def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
#         self.max_hands = max_hands
#         self.detection_confidence = detection_confidence
#         self.tracking_confidence = tracking_confidence
#
#         # Initialize Mediapipe Hand solutions
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             max_num_hands=self.max_hands,
#             min_detection_confidence=self.detection_confidence,
#             min_tracking_confidence=self.tracking_confidence
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#
#     def find_hands(self, image, draw=True):
#         # Convert the BGR image to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(image_rgb)
#
#         # Draw landmarks on the hand if draw is True
#         if self.results.multi_hand_landmarks and draw:
#             for hand_landmarks in self.results.multi_hand_landmarks:
#                 self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
#
#         return image
#
#     def get_landmarks(self, image):
#         landmarks = []
#         if self.results.multi_hand_landmarks:
#             for hand_landmarks in self.results.multi_hand_landmarks:
#                 for idx, lm in enumerate(hand_landmarks.landmark):
#                     # Get height, width of the image
#                     h, w, _ = image.shape
#                     # Convert landmark position to pixel position
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     landmarks.append((cx, cy))
#         return landmarks
#
#     def fingers_up(self):
#         fingers = []
#         if self.results.multi_hand_landmarks:
#             hand_landmarks = self.results.multi_hand_landmarks[0].landmark
#
#             # Thumb: Tip (4) compared to the IP joint (3)
#             if hand_landmarks[4].x < hand_landmarks[3].x:  # Right hand logic, adjust for left
#                 fingers.append(1)  # Thumb is open
#             else:
#                 fingers.append(0)  # Thumb is closed
#
#             # Other fingers: Compare tip (8, 12, 16, 20) with PIP joint (6, 10, 14, 18)
#             finger_tips = [8, 12, 16, 20]
#             pip_joints = [6, 10, 14, 18]
#
#             for i in range(4):
#                 if hand_landmarks[finger_tips[i]].y < hand_landmarks[pip_joints[i]].y:
#                     fingers.append(1)  # Finger is open
#                 else:
#                     fingers.append(0)  # Finger is closed
#
#         return fingers
#
#
# # Main program for hand tracking with fingers detection
# def main():
#     cap = cv2.VideoCapture(0)
#     tracker = HandTrackingModule(max_hands=1)
#
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty frame.")
#             continue
#
#         # Flip the image and find hands
#         image = cv2.flip(image, 1)
#         image = tracker.find_hands(image)
#
#         # Get the list of open/closed fingers
#         finger_state = tracker.fingers_up()
#         if finger_state==[1,1,1,1,1]:
#             print('open all')
#         elif finger_state==[0,0,0,0,0]:
#             print('close all')
#
#         # Display the finger state on the image
#         cv2.putText(image, str(finger_state), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#
#         # Show the output
#         cv2.imshow('Hand Tracking', image)
#
#         # Exit on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()