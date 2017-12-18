# -*- coding: utf-8 -*-
#"""
#Created on Thu Dec 14 18:53:51 2017

#@author: shahb
#"""
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import os
import numpy as np


subjects = ["", "User1", "User2"]


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def blink(frame):
    
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # Grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
	# detect faces in the grayscale frame
    rects = detector(gray, 0)
   # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        #cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return ear
#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    #load OpenCV face detector, I am using LBP which is fast
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None,0
 
    #extract the face area
    (x, y, w, h) = faces[0]
 
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0],1




#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
 
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
 
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
 
    #let's go through each directory and read images within it
    for dir_name in dirs:
 
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;    
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
 
        #build path of directory containing images for current subject subject
        subject_dir_path = data_folder_path + "/" + dir_name
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
    #go through each image name, read image, 
    #detect face and add face to list of faces

        for image_name in subject_images_names:
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            #build image path
            image_path = subject_dir_path + "/" + image_name
            #read image
            image = cv2.imread(image_path)
            #detect face
            face, rect,i = detect_face(image)
 
         #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
 
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.destroyAllWindows()
    return faces, labels

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and height
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
 


#this function recognizes the person in image passed
 #and draws a rectangle around detected face with name of the subject
def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    face, rect,i = detect_face(img)
    #if face is detected
    if(i==1):
        #predict the image using our face recognizer 
        label= face_recognizer.predict(face)
        #calculation of eye aspect ratio
        ear = blink(img)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
 
    if(i==1):
        return img,i,ear,rect,label_text
    else:
        return img,i,None,None,None
        #if face is not detected then return image only



if __name__ == '__main__':
    #let's first prepare our training data
    #data will be in two lists of same size
    #one list will contain all the faces
    #and the other list will contain respective labels for each face
    print("Preparing data...")
    faces, labels = prepare_training_data('Training_Data_Location')
    print("Data prepared")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    #Threshold eye aspect ratio is set to 0.25
    EYE_AR_THRESH = 0.25
    #create our LBPH face recognizer 
    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
    #for input from camera
    cam=cv2.VideoCapture(0)
    #initial time set to both variable
    noBlinking_time=time.time()
    blinking_time=time.time()
    while(1):
        ret, test_img1 = cam.read()
        #perform a prediction
        predicted_img1,i,ear,rect,label_text = predict(test_img1)
        if(i==1):
            if ear < EYE_AR_THRESH:
                #Time recorder when Blinking of eye took place
                blinking_time=time.time()
            else:
                #Time at which No blinking of eye was there
                noBlinking_time = time.time()
            #Time gap in between a blink should take place so that it is live
            if((noBlinking_time-blinking_time)<3):
                label_text = label_text+" Live"
                #print(noBlinking_time-blinking_time)
            else:
                label_text = label_text+" Photo"
            draw_text(predicted_img1, label_text, rect[0], rect[1]-5)
        cv2.imshow(subjects[1], predicted_img1)
        if (cv2.waitKey(32) & 0xff) == 27:
            break
    cv2.destroyAllWindows()
    cam.release()
