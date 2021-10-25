# This is the final product for the project.
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
left = [23,24,25,26,27,28,29,30,31,253,254,255,256,257,258,259,260,261,185,40,39,0,270,269,409,375,321,405,314,17,84,181,91,146]
dict1 = {23:'',24:'',25:'',26:'',27:'',28:'',29:'',30:'',31:'',253:'',254:'',255:'',256:'',257:'',258:'',259:'',260:'',261:'',185:'',40:'',39:'',0:'',270:'',269:'',409:'',375:'',321:'',405:'',314:'',17:'',84:'',181:'',91:'',146:''}
count = 0
loaded_model = pickle.load(open('Drowsiness_model.h5', 'rb'))
print('done')
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        c = -1
        l = [61, 185, 40, 39, 37, 0, 270,269, 409, 291,375, 321, 405, 314, 17, 84, 181, 91,146, 61]
        for face in results.multi_face_landmarks:
          for landmark in face.landmark:
            c+=1
            if (c>=23 and c<=31) or (c>=253 and c<=261) or(c in [185, 40, 39, 0, 270,269,409,375, 321,405, 314, 17,84,181, 91,146]):
              if c in left:
                x = landmark.x
                y = landmark.y
                z = landmark.z

                shape = image.shape 
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                cv2.circle(image, (relative_x, relative_y), radius=5, color=(225, 0, 100), thickness=1)  
                dict1[c]=[x,y,z]
          if cv2.waitKey(5)==ord('q'):
            count+=1
            l = list(list(dict1.values()))
            # print(l)
            new_df = {}
            for key,item in dict1.items():
              # print(key,item)
              new_df[str(key)+'_x'] = item[0]
              new_df[str(key)+'_y'] = item[1]
              new_df[str(key)+'_z'] = item[2]
            # l = np.array(l,dtype='object')
            # l = l.reshape(-1,1)
            # l = l.tolist()
            df = pd.DataFrame([new_df])
            # df.set_index(inplace=True)
            # print(df)
            # df = df.loc[:,df.columns!=['Status']]
            y_pred = loaded_model.predict(df)
            if y_pred[0]==1:
              print('Drowsy')
            else:
              print('Not Drowsy')
            # print(y_pred)
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()    
