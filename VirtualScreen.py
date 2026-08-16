import cv2
import mediapipe as mp
import time
import numpy as np

BaseOptions = mp.tasks.BaseOptions
Handlandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

hand_landmarker = Handlandmarker.create_from_options(options)

#kamera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("kamera gabisa dibuka")
    exit()
    
start_time = time.time()

#loop

while cam.isOpened():
    ret, frame = cam.read()
    
    
    if not ret:
        print("gagal baca frame")
        break
    
    #convert BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #bikin media pipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )
    
    #timestamp
    timestamp_ms = int((time.time() - start_time) * 1000)
    
    #detect tangan
    result = hand_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )
    
    height, width, _ = frame.shape
    left_index = None
    left_thumb = None
    right_index = None
    right_thumb = None
    
    
    
    if result.hand_landmarks:
        print("jumlah tangan terdeteksi:", len(result.hand_landmarks))
        
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            #info kiri/kanan
            handedness = result.handedness[i][0]
            hand_name = handedness.category_name
            
            #ujung jempol dan telunjuk
            thumb_tip = hand_landmarks[4]
            index_finger_tip = hand_landmarks[8]
            
            thumb_tip_x = int(thumb_tip.x * width)
            thumb_tip_y = int(thumb_tip.y * height)
            
            index_finger_tip_x = int(index_finger_tip.x * width)
            index_finger_tip_y = int(index_finger_tip.y * height)
            
            if hand_name == "Left":
                left_index = (index_finger_tip_x, index_finger_tip_y)
                left_thumb = (thumb_tip_x, thumb_tip_y)
                
            elif hand_name == "Right":
                right_index = (index_finger_tip_x, index_finger_tip_y)
                right_thumb = (thumb_tip_x, thumb_tip_y)
                
        if left_index is not None and left_thumb is not None and right_index is not None and right_thumb is not None:
            overlay = frame.copy()
            
            points = np.array([
                    left_index,
                    right_index,
                    right_thumb,
                    left_thumb
                ], dtype=np.int32)
            
            mask = np.zeros((height, width), dtype=np.uint8)
                
            cv2.fillPoly(
                    mask,
                    [points],
                    255
                )
            
            blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
            blurred_area = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
            
            mask_inv = cv2.bitwise_not(mask)
            normal_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
            
            frame = cv2.add(normal_area, blurred_area)
            
            #gambar garisnya
            
            cv2.line(
                frame,
                left_index,
                right_index,
                (0,0,255),
                3
            )
            cv2.line(
                frame,
                right_index,
                right_thumb,
                (0,0,255),
                3
            )
            cv2.line(
                frame,
                right_thumb,
                left_thumb,
                (0,0,255),
                3
            )
            
            
            cv2.circle(
                frame,
                (thumb_tip_x, thumb_tip_y),
                10,
                (0, 0, 255),
                -1
            )
            
            cv2.circle(
                frame,
                (index_finger_tip_x, index_finger_tip_y),
                10,
                (0, 0, 255),
                -1
            )
            
    if left_index is not None:
        cv2.circle(
            frame,
            left_index,
            10,
            (0, 0, 255),
            -1
        )
    if left_thumb is not None:
        cv2.circle(
            frame,
            left_thumb,
            10,
            (0, 0, 255),
            -1
        )
    if right_index is not None:
        cv2.circle(
            frame,
            right_index,
            10,
            (0, 0, 255),
            -1
        )
    if right_thumb is not None:
        cv2.circle(
            frame,
            right_thumb,
            10,
            (0, 0, 255),
            -1
        )
        
    if (left_index is not None and left_thumb is not None and right_index is not None and right_thumb is not None):
        cv2.line(
            frame,
            left_index,
            right_index,
            (0,0,255),
            3
        )
        cv2.line(
            frame,
            left_thumb,
            right_thumb,
            (0,0,255),
            3
        )
        cv2.line(
            frame,
            left_index,
            left_thumb,
            (0,0,255),
            3
        )
        cv2.line(
            frame,
            right_index,
            right_thumb,
            (0,0,255),
            3
        )

    cv2.imshow("Hand Landmarker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
hand_landmarker.close()
cv2.destroyAllWindows()
