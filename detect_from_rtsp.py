import numpy as np
import cv2
import face_recognition
import os
import threading

# Create bufferless vid capture object
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        ret, frame = self.cap.retrieve()
        return frame

# Load trained images
trained_path = "./trained_images/"
trained_images = []
trained_images_names = []

# Path to place unkown faces
unkown_path = "./unkown_faces/"

# Loop through all image paths in trained images folder
for image_path in os.listdir(trained_path):

  # Create full input path and read then encode image
  input_path = os.path.join(trained_path, image_path)
  image = cv2.imread(input_path)
  encoded = face_recognition.face_encodings(image)[0]
  trained_images.append(encoded)

  # Add name to face
  name = os.path.splitext(image_path)[0]
  trained_images_names.append(name)

# Get rtsp stream
vid = VideoCapture("rtsp://192.168.1.86:8554/cam")

unkown_count = 1
while(True):

  face_names = []

  # Capture vid frame by frame
  frame = vid.read()

  # If frames been read ok
  if frame is not None: 

    # Find location of faces in frame, encode those
    unknown_face_locations = face_recognition.face_locations(frame)
    unkown_image_encoding = face_recognition.face_encodings(frame, unknown_face_locations)

    # Iterate through encoded faces in unkown image
    for face_encoding in unkown_image_encoding:

      # Find if face is a match to known faces
      print("unkown matched")
      matches = face_recognition.compare_faces(trained_images, face_encoding)
      name = "unkown"

      # Find known face with shortest distance to unkown one
      face_distances = face_recognition.face_distance(trained_images, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = trained_images_names[best_match_index]

      # Append name of face (either found or unkown) to array
      face_names.append(name)

      # Draw name and bounding boxes onto image
      for (top, right, bottom, left), name in zip(unknown_face_locations, face_names):

        if name == "unkown":

          # Crop image to just face
          crop_face = frame[top: bottom, left: right]

          # Encode unknown face
          unkown_face_encoded = face_recognition.face_encodings(crop_face)

          # If high enough conf there is a face, grab first face in array
          if unkown_face_encoded:

            unkown_face_encoded = unkown_face_encoded[0]

            # Get file name of unkown face
            name = name + str(unkown_count)
            unkown_face_name = name + '.jpg'
            unkown_count += 1

            # Encode face
            trained_images.append(unkown_face_encoded)
            trained_images_names.append(unkown_face_name)

            # Save face to file system
            unkown_face_path = os.path.join(unkown_path, unkown_face_name)
            cv2.imwrite(unkown_face_path, crop_face)

            print("Unkown face detected: " + unkown_face_name)

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label with name below corrosponding face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)

    # Display image w/ bounding boxes over faces
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# After the loop release the cap object
vid.release()
cv2.destroyAllWindows()