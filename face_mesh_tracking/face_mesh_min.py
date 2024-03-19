import cv2
import mediapipe as mp
import time

vid_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/FacialVideos/"
cap = cv2.VideoCapture(vid_path + "video1.mp4")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles


prev_time = 0

while True:
    success, img = cap.read()

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process RGB img with facemesh
    results = facemesh.process(img_rgb)
    print(results)

    
    if results.multi_face_landmarks:
        
        # Draw face mesh
        for facelm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelm, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, mp_drawing_styles.get_default_face_mesh_contours_style())

            # Print mesh landmark information 
            for id, lm in enumerate(facelm.landmark):
                # print(lm)

                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print ("ID:", id, " x:", x, " y:", y)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f"FPS counter: {int(fps)}", (50, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 200, 0), 1)

    cv2.imshow('Face detection module', img)

    cv2.waitKey(60)
