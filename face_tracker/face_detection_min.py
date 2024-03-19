import cv2
import mediapipe as mp
import time

#Load videos
video_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/FacialVideos/"
cap = cv2.VideoCapture(video_path + "video1.mp4")
previous_time = 0

# Declaring which libraries we'll be using
faceDetection = mp.solutions.face_detection.FaceDetection()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    if not success:
        break

    # Converts video from BGR to RGB and stores it in results
    image_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(image_RGB)
    print(results)
    

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)

            # Detection confidence
            detection_confidence = detection.score
            print('Detection confidence: ' + str(detection_confidence))

            # Bounding box locations
            bBox_ratio = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bBox = int(bBox_ratio.xmin * iw), int(bBox_ratio.ymin * ih), \
                    int(bBox_ratio.width * iw), int(bBox_ratio.height * ih)
            
            # Draw a rectange around bbox locations
            cv2.rectangle(img, bBox, (255, 0, 255), 2)

            # Display detection confidence as percentage above top 2 bbox points
            cv2.putText(img, f'Confidence: {int(detection_confidence[0] * 100)}%', (bBox[0], bBox[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 200), 1)
            
          
            
            print('Location of bounding box: \n' + str(bBox))

    

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, "FPS counter: " + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)


    cv2.imshow('Facial recognition video', img)
    cv2.waitKey(65)