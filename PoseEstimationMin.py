import cv2
import time
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("PoseVideos/2.mp4")
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, shape = img.shape
            x_pixel, y_pixel = int(width * lm.x), int(height * lm.y)
            cv2.circle(img, (x_pixel, y_pixel), 5, (255, 0, 255), cv2.FILLED)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
