import cv2
import mediapipe as mp
import math
import numpy as np
from fontTools.misc.cython import returns
from google.protobuf.json_format import MessageToDict
from math import hypot
from brightnes_lefthand import Brightness##for Bright ness control
##for volume control
from volume_control_righthand import Volume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
from zoominout import ZoomInOut
from virtualmouse import VirtualMouse
devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
volbar=400
volper=0
volMin,volMax=volume.GetVolumeRange()[:2]

mphands=mp.solutions.hands
hands=mphands.Hands(static_image_mode=False,
                    model_complexity=1
                    ,max_num_hands=2,
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.5)
Draw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
while True:
    _,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        if len(results.multi_handedness)==2:
            # cv2.putText(img,'bothhands',
            #             (250,50),
            #             cv2.FONT_HERSHEY_COMPLEX,
            #             1,
            #             (0,255,0),
            #             2
            #             )
            # cv2.imshow('img',img)
            ZoomInOut(img, imgRGB, results, Draw, mphands, hands)
        else:
            for i in results.multi_handedness:
                label=MessageToDict(i)['classification'][0]['label']
                if label=='Left':
                    # cv2.putText(img,label+'hand',(250,50),
                    #             cv2.FONT_HERSHEY_COMPLEX,
                    #             1,(0,255,255),
                    #             2)
                    Brightness(img,imgRGB,results,Draw,mphands,hands)
                if label=='Right':
                    # cv2.putText(img,label+'hand',(460,50),
                    #             cv2.FONT_HERSHEY_COMPLEX,
                    #             1,(0,255,0),2)

                    cv2.putText(img, label + ' hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # Extract landmark points for the right hand
                    landmarks = hand_landmarks.landmark
                    index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
                    index_dip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP]
                    thumb_ip = landmarks[mp.solutions.hands.HandLandmark.THUMB_IP]

                    # Calculate distances to detect gestures
                    index_thumb_distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5
                    index_dip_distance = ((index_tip.x - index_dip.x) ** 2 + (index_tip.y - index_dip.y) ** 2) ** 0.5

                    # Thresholds for gesture detection (tuned as needed)
                    index_open_threshold = 0.05
                    index_thumb_open_threshold = 0.1

                    # Check if index finger is open (for virtual mouse control)
                    if index_dip_distance > index_open_threshold and index_thumb_distance > index_thumb_open_threshold:
                        # If only the index finger is open, enable virtual mouse control
                        # Call the function for virtual mouse operation (replace `VirtualMouse` with your function)
                        VirtualMouse(img, imgRGB, results, Draw, mphands, hands)
                        cv2.putText(img, 'Virtual Mouse Active', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),
                                    2)

                    # Check if index finger and thumb are both open (for volume control)
                    if index_dip_distance > index_open_threshold and index_thumb_distance < index_thumb_open_threshold:
                        # If both the index finger and thumb are open, enable volume control
                        Volume(img, imgRGB, results, Draw, mphands, hands)
                        cv2.putText(img, 'Volume Control Active', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                                    2)

    # cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()