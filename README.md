# Live-PhotoDifferentiation
This repository contains a python code which recognise a person and differentiate it whether real or photo of that person.

# About Code
First training of face data is done. All the face data is stored in training data folder. This training data folder contains a number of images of different people stored in respective file.
On the basis of trained data, face recognition is done.
On the recognised face, live-photo differentiator is run.

# Header files include
*scipy
*imutils
*time
*dlib
*cv2
*os
*numpy

# Changes need to be done before running code
In the line 17 of the code, change the name of persons whose image is present in training data
In the line 206,  change the location of training data
In the line 210, change the location of 'shape_predictor_68_face_landmarks.dat' file
In the line 219, change the type of input method

