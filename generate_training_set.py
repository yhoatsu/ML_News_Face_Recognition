import os
import pickle
import dlib
import sys

# To check if the params are well introduced by the user
if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./generate_training_set.py 5_face_landmarks hog ./images/training \n"
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
        "of calculated the 128D array that defined that face.")
    exit()

# Save at variables the methods and path introduced by the user
predictor_model = sys.argv[1]
face_rec_model = sys.argv[2]
path_object = sys.argv[3]
resample_times = int(sys.argv[4])

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

# To save in a variable the path where that script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define two empty lists to save there the 128D array that define each face and the label associated to it
y_labels = []
x_train = []

# At this loop we go though all the images located at the training folder
for root, dirs, files in os.walk(path_object):
    for file in files:
        # Check if a file is an image or not
        if file.endswith("jpeg") or file.endswith("jpg"):
            # Save the complete path to the image to load it later
            path = os.path.join(root, file)
            # Save as label the name of the folder that contain the images and to replace the white spaces to _
            label = os.path.basename(root).replace(" ", "_")

            # Read the image as a numpy array
            image = dlib.load_rgb_image(path)

            # Save the detects faces at the variable faces using the detector defined before
            faces = face_detector(image, 1)

            # That loop is analyze each face detected at the variable faces
            for face in faces:
                # The detector return different objects depend if we use cnn or hog
                if face_rec_model == "cnn":
                    # Here we get the landmarks that defines the face.
                    shape = shape_pred(image, face.rect)
                else:
                    shape = shape_pred(image, face)

                # Computes the 128D array that define the face and that'll be used to predict the label of the face.)
                face_descriptor = face_rec.compute_face_descriptor(image, shape, resample_times)

                # Add to the lists x_train y y_labels the 128D array that describe the face and the label associated.
                x_train.append(face_descriptor)
                y_labels.append(label)

# Save in a pickle object the lists x_train and y_labels
# adding information at the file about the predictor and face recognition model used

if 'pickles' not in os.listdir(BASE_DIR):
    os.makedirs(os.path.join(BASE_DIR, "pickles"))

with open(os.path.join(BASE_DIR, "pickles/"+"x_train_"+predictor_model+"_"+face_rec_model+".pickle"), 'wb') as f:
    pickle.dump(x_train, f)
with open(os.path.join(BASE_DIR, "pickles/"+"y_labels"+predictor_model+"_"+face_rec_model+".pickle"), 'wb') as f:
    pickle.dump(y_labels, f)
