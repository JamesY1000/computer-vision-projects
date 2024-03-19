import cv2
import mediapipe as mp
import time



mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles

# Class which has 2 methods + initialisation - 1 for drawing and connecting images, 2 for 
# finding and displaying specific landmark locations
class faceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mpDraw
        self.mpFaceMesh = mpFaceMesh
        self.facemesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks,
                                                 self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_drawing_styles = mp_drawing_styles


    def detectMesh(self, img, draw=True):

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process RGB img with facemesh
        self.results = self.facemesh.process(img_rgb)
        # print(self.results)

        x, y = 0, 0
        
        # Stores all faces with their information
        faces = []

        if self.results.multi_face_landmarks:
            

            # Draw face mesh
            for facelm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelm, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, mp_drawing_styles.get_default_face_mesh_contours_style())

                # Stores lm information of a single face
                face = []

                # Print mesh landmark information 
                for id, lm in enumerate(facelm.landmark):
                    # print(lm)

                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print ("ID:", id, " x:", x, " y:", y)
                    face.append([id, x, y])
                faces.append(face)

        return img, faces


def main():
    vid_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/FacialVideos/"
    cap = cv2.VideoCapture(vid_path + "video1.mp4")
    prev_time = 0
    detector = faceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        img, faces = detector.detectMesh(img, draw=True)

        # Print out data of first landmark of first face
        if len(faces) != 0:
            print("id:", faces[0][0][0], " x:", faces[0][0][1], " y:", faces[0][0][2])

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"FPS counter: {int(fps)}", (50, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 200, 0), 1)

        cv2.imshow('Face detection module', img)

        cv2.waitKey(60)

if __name__ == "__main__":
    main()