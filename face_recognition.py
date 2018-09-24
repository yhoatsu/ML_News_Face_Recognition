########################################################################################################################
#
# The contents of this file are in the public domain.
#
#   This script is a test version of face recognition dedicated to recognize
#   politics and journalists at the public channel TV in Spain.
#   That project is inmerse within the context of end of master of Bid Data
#   Analytics project.
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
# COMPILING/INSTALLING THE face_recognition PYTHON INTERFACE
#   You can install face_recognition using the command (you need to install dlib before):
#       pip install face_recognition
#
# COMPILING/INSTALLING THE OpenCV PYTHON INTERFACE
#   You can install OpenCV using the command (you need to install dlib before):
#       pip install cv2
#
# DESCRIPTIONS of the libraries used in that script
# Treatment for video and images


# That library allow us to obtain the frame in a video, add layers to the frame, save the results as a video, etc.
import cv2
# To save and load python object
import pickle
# To optimize math operations and add news ones to the base version of Python
import numpy as np
# To obtain the path of images, videos, models and results
import os
# To allow to introduce arguments through shell console
import sys
# Library to detect faces, read the models to obtain the features of a face and to obtain the pixels where the face is.
import dlib

# To save images images/frames or video, we save the base path where that script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# To check if the params are well introduced by the user
if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py 5_face_landmarks hog ./images/test.jpg \n"
        "----------------------------------------------------------\n"
        "The first param is about which face recognition model to use (5_face_landmarks or 68_face_landmarks),\n"
        "5_face_landmarks is less accurate but faster and that model extract only 5 point to identify a face.\n"
        "68_face_landmarks is more accurate but slower and that model extract 68 point to identify a face.\n"
        "----------------------------------------------------------\n"
        "The second param is about which face detection model to use (hog or cnn),\n"
        "hog is less accurate but faster on CPUs. cnn is a more accurate \n"
        "deep-learning model which is GPU/CUDA accelerated (if available). The default is hog.\n"
        "----------------------------------------------------------\n"
        "The third param is the path where the folders with the images to use as training set are located.\n"
        "----------------------------------------------------------\n"
        "The fourth param is the times that the face is re-sampled and slightly modified each time at the moment\n"
        "of calculated the 128D array that defined that face.\n")
    exit()

# Save at variables the methods and path introduced by the user
predictor_model = sys.argv[1]
face_rec_model = sys.argv[2]
path_object = sys.argv[3]
resample_times = int(sys.argv[4])

# At the script "generate_training_set" we save two items:
# x_train -> that is a list with the features of each face in our database that we'll use to predict.
# Depend of the model that features will be an 128D numpy array or 5D numpy array
# y_labels -> that is a list containing the labels associate to the features of the faces used as training
# So the lines bellow load that two items.
with open("pickles/"+"x_train_"+predictor_model+"_"+face_rec_model+".pickle", 'rb') as f:
    x_train = pickle.load(f)
with open("pickles/"+"y_labels"+predictor_model+"_"+face_rec_model+".pickle", 'rb') as f:
    y_labels = pickle.load(f)

# Load a shape predictor to find face landmarks so we can precisely localize the face.
if predictor_model == '5_face_landmarks':
    shape_pred = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
elif predictor_model == '68_face_landmarks':
    shape_pred = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
else:
    print("Error, face detection model not valid, select 5_face_landmarks or 68_face_landmarks.")
    exit()

# Load a face detection model to obtain the rectangle around the face.
if face_rec_model == 'cnn':
    face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
elif face_rec_model == 'hog':
    face_detector = dlib.get_frontal_face_detector()
else:
    print("Error, face detection model not valid, select 5_face_landmarks or 68_face_landmarks.")
    exit()

# Load a face recognition model that generate a 128D array to classify the face.
face_rec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


# Model K-NN to classify a face_descriptor using the training_set obtained with the script generate_training_set.py
def my_knn(face_descriptor_to_predict, training_set=x_train, labels_training=y_labels,
           neighborhoods=3, ord_norm=2, threshold=0.475):
    # Calculates the L^n norm for each pair of values training_set[q]-face_descriptor_to_predict
    # By defect calculates L2 norm (euclidean distance)
    distances = np.linalg.norm(training_set - face_descriptor_to_predict, axis=1, ord=ord_norm)
    # To order the array of distances and save the arguments or locations of the values of distances
    locations_lower_distances = np.argsort(distances)

    # If the min distance between the introduced face descriptor and the faces at the training set
    # aren't to close (defined by the threshold) then stop here and return as Unknown the face under analysis
    if distances[locations_lower_distances[0]] < threshold:
        print(distances[locations_lower_distances[0]])
        # To get the labels of the K nearest neighborhoods
        labels_k_neighborhoods = [labels_training[index] for index in locations_lower_distances[:neighborhoods]]
        # To count the number of labels that appears repeated
        label, count = np.unique(labels_k_neighborhoods, return_counts=True)
        # To create a dictionary to save the counts obtained at the line above
        dictionary_counts_labels_knn = dict(zip(label, count))

        # Save at the variable prediction the label that appear more than the rest at the k-nn
        prediction = max(dictionary_counts_labels_knn, key=dictionary_counts_labels_knn.get)
        print(prediction)
    else:
        prediction = 'Unknown'

    # Return the label with the prediction of the face
    return prediction


# Save at a variable the termination of the object introduced by the user to analyze,
# to know if is a image or a video (same as .endswith(""))
type_object = path_object.split(".")[-1]

# Face detection in one image
if type_object == 'jpg' or type_object == 'jpeg':
    # To define a blank window where we'll draw an numpy image
    win = dlib.image_window()

    # Read the image of the user as a numpy array
    image = dlib.load_rgb_image(path_object)
    # Save the detects faces at the variable faces using the detector defined before
    faces = face_detector(image, 1)

    # To print at the blank window the image loaded
    win.set_image(image)

    # That loop is analyze each face detected at the variable faces
    for face in faces:
        # Remove all overlays from the image_window.
        win.clear_overlay()

        # The detector return different objects depend if we use cnn or hog
        if face_rec_model == "cnn":
            # Here we get the landmarks that defines the face.
            shape = shape_pred(image, face.rect)
            # Add a layer at the image showing the box where the face is.
            win.add_overlay(face.rect)
        else:
            shape = shape_pred(image, face)
            win.add_overlay(face)

        # Add a layer at the image showing the landmarks that defines the face.
        win.add_overlay(shape)

        # Computes the 128D array that define the face and that'll be used to predict the label of the face.
        face_descriptor = np.array(face_rec.compute_face_descriptor(image, shape))

        # distances = np.linalg.norm(face_encodings - face_to_compare, axis=1)
        label_predicted = my_knn(face_descriptor, neighborhoods=1)
        print(label_predicted)

        # Wait util the user press the Enter key to continue.
        dlib.hit_enter_to_continue()

elif type_object == 'mp4' or type_object == 'avi':

    video_in = cv2.VideoCapture(path_object)
    video_out = cv2.VideoWriter('video_result.avi', cv2.VideoWriter_fourcc(*'XVID'),
                                video_in.get(5), (int(video_in.get(3)), int(video_in.get(4))))

    while video_in.isOpened():

        ret, frame = video_in.read()

        faces = face_detector(frame, 1)

        for face in faces:
            # The detector return different objects depend if we use cnn or hog
            if face_rec_model == "cnn":
                # Here we get the landmarks that defines the face.
                shape = shape_pred(frame, face.rect)
                top, right, bottom, left = face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()
            else:
                shape = shape_pred(frame, face)
                top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()

            # Computes the 128D array that define the face and that'll be used to predict the label of the face.
            face_descriptor = np.array(face_rec.compute_face_descriptor(frame, shape, resample_times))

            label_predicted = my_knn(face_descriptor, neighborhoods=1)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, str(label_predicted), (left, top),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        video_out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()

