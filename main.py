import face_recognition
import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import numpy as np
import csv
import os

from datetime import datetime


#---------------for date and time----------------------

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

#_____________________open web cam______________
video_capture = cv2.VideoCapture(0)


video = VideoWriter(current_date+'.avi', VideoWriter_fourcc(*'MP42'), 05.0, (640,480)  )

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#__________________Providing the images from the folder for face encodings______________________


elon_img = face_recognition.load_image_file("images/elon.jpg")
elon_encodings = face_recognition.face_encodings(elon_img)[0]

ratan_img = face_recognition.load_image_file("images/ratan.jpg")
ratan_encodings = face_recognition.face_encodings(ratan_img)[0]

nirmala_img = face_recognition.load_image_file("images/nirmala.jpg")
nirmala_encodings = face_recognition.face_encodings(nirmala_img)[0]

mark_img = face_recognition.load_image_file("images/mark.jpg")
mark_encodings = face_recognition.face_encodings(mark_img)[0]

known_face_encoding = [elon_encodings, ratan_encodings, nirmala_encodings, mark_encodings ]

known_face_names = ["Elon Musk", "Ratan Tata", "Nirmala Sitaraman", " Mark Zuckerberg"]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True


#-----------------for csv file--------------------

f = open(current_date+'.csv', 'w+', newline= '')
lnwriter = csv.writer(f)   # class instance

#_______________infinite loop_____________________ takin video input
while True:
    ret,frame = video_capture.read()

    print(ret)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = cascade_face.detectMultiScale(g, 1.3, 4)

    small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy = 0.75)
    rgb_small_frame = small_frame[:, :, ::-1]
    #_______________recognize the face names__________________
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for (x, y, w, hi) in h:
            cv2.rectangle(frame, (x, y), (x + w, y + hi), (0, 255, 0), 4)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)



                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    # Display the names of the recognised face


                # __________________saving the names in csv file_____________________
                face_names.append(name)
                if name in known_face_names:
                    if name in students:
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (x + 6, y - 6), font, 1.0, (0, 0, 255), 4)
                        students.remove(name)
                        print(students)
                        current_time = now.strftime(" %I - %M - %p ")

                        lnwriter.writerow([name, current_time])

    cv2.imshow("Attendence System", frame)

    video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # exit when we press q button
       break

video_capture.release()

video.release()
cv2.destroyAllWindows()
f.close()