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
#
import time
from datetime import date
#
import collections
#
import pandas as pd

# To save images images/frames or video, we save the base path where that script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# To check if the params are well introduced by the user
if len(sys.argv) != 7:
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
        "The third param is the complete path to the image or video to analyze.\n"
        "----------------------------------------------------------\n"
        "The fourth param is the times that the face is re-sampled and slightly modified each time at the moment\n"
        "of calculated the 128D array that defined that face.\n"
        "----------------------------------------------------------\n"
        "The fifth param is to show the images/video with the marks of face recognition and also the landmarks\n"
        "od the face if the object of analysis is a image. So, write 'yes' to see it or 'no' in other case.\n"
        "At the case of a video analysis, to stop the analysis you can push 'q' while you watching the video result\n"
        "----------------------------------------------------------\n"
        "The sixth param is to change the threshold of recognition, you can choose any number between 0 and 1\n"
        "if the number id near to 1 (0.9 for example) you will have to many false positive, but if you enter\n"
        "a number near to 0 you'll be probably don't get any positive match. The defect value is 0.475.")
    exit()

# Save at variables the methods and path introduced by the user
predictor_model = sys.argv[1]
face_rec_model = sys.argv[2]
path_object = sys.argv[3]
resample_times = int(sys.argv[4])
verbose = sys.argv[5]
threshold_user = float(sys.argv[6])

# At the script "generate_training_set" we save two items:
# x_train -> that is a list with the features of each face in our database that we'll use to predict.
# Depend of the model that features will be an 128D numpy array or 5D numpy array
# y_labels -> that is a list containing the labels associate to the features of the faces used as training
# So the lines bellow load that two items.
with open("pickles/"+"x_train_"+predictor_model+"_"+face_rec_model+".pickle", 'rb') as f:
    x_train = pickle.load(f)
with open("pickles/"+"y_labels"+predictor_model+"_"+face_rec_model+".pickle", 'rb') as f:
    y_labels = pickle.load(f)
with open("pickles/metadata_faces_recognized.pickle", 'rb') as f:
    dictionary_faces = pickle.load(f)

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
def my_knn(face_descriptor_to_predict, coef=0, training_set=x_train, labels_training=y_labels,
           neighborhoods=3, ord_norm=2, threshold=threshold_user):
    # Calculates the L^n norm for each pair of values training_set[q]-face_descriptor_to_predict
    # By defect calculates L2 norm (euclidean distance)
    distances = np.linalg.norm(training_set - face_descriptor_to_predict, axis=1, ord=ord_norm) + coef
    # To order the array of distances and save the arguments or locations of the values of distances
    locations_lower_distances = np.argsort(distances)

    # If the min distance between the introduced face descriptor and the faces at the training set
    # aren't to close (defined by the threshold) then stop here and return as Unknown the face under analysis
    if distances[locations_lower_distances[0]] < threshold:

        # To get the labels of the K nearest neighborhoods
        labels_k_neighborhoods = [labels_training[index] for index in locations_lower_distances[:neighborhoods]]
        # To count the number of labels that appears repeated
        label, count = np.unique(labels_k_neighborhoods, return_counts=True)
        # To create a dictionary to save the counts obtained at the line above
        dictionary_counts_labels_knn = dict(zip(label, count))

        # Save at the variable prediction the label that appear more than the rest at the k-nn
        prediction = max(dictionary_counts_labels_knn, key=dictionary_counts_labels_knn.get)

    else:
        prediction = 'Unknown'

    # Return the label with the prediction of the face
    return prediction


#
def predict_face(labels):
    count_rep_elements = collections.Counter(labels)
    del count_rep_elements['Unknown']
    dict_count_rep_elements = count_rep_elements.most_common(1)
    if len(dict_count_rep_elements) == 0:
        most_freq_element = 'Unknown'
    elif dict_count_rep_elements[0][1] < 10:
        most_freq_element = 'Unknown'
    else:
        most_freq_element = dict_count_rep_elements[0][0]
    return most_freq_element


def frame_to_time(frame):
    time_raw = frame/25
    time_milli_secs = int((time_raw - int(time_raw))*100)
    time_secs_acum = int(time_raw)
    time_min = int(time_secs_acum/60)
    time_hours = int(time_min/60)
    time_secs = time_secs_acum - time_min*60
    return str(time_hours) + ':' + str(time_min) + ':' + str(time_secs) + '.' + str(time_milli_secs)


# Save at a variable the termination of the object introduced by the user to analyze,
# to know if is a image or a video (same as .endswith(""))
type_object = path_object.split(".")[-1]

if type_object == 'mp4' or type_object == 'avi':

    video_in = cv2.VideoCapture(path_object)
    video_out = cv2.VideoWriter('video_result_relabeled.avi', cv2.VideoWriter_fourcc(*'XVID'),
                                min(25.0, video_in.get(5)), (int(video_in.get(3)), int(video_in.get(4))))
    num_frame = 0
    while video_in.isOpened():
        num_frame += 1
        ret, frame = video_in.read()
        for faces_in_dict_final in dictionary_faces.values():
            if num_frame in faces_in_dict_final['num_frames']:
                pos_in_dictionary = faces_in_dict_final['num_frames'].index(num_frame)
                label_predicted = faces_in_dict_final['label_predicted']
                left = faces_in_dict_final['position'][pos_in_dictionary]['left']
                top = faces_in_dict_final['position'][pos_in_dictionary]['top']
                right = faces_in_dict_final['position'][pos_in_dictionary]['right']
                bottom = faces_in_dict_final['position'][pos_in_dictionary]['bottom']
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
                cv2.putText(frame, label_predicted, (left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if verbose == "yes":
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_out.write(frame)

        if num_frame == int(video_in.get(7)):
            break

    # When everything done, release the capture
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()

    start_frames = []
    end_frames = []
    labels = []
    planes = []

    for faces in dictionary_faces.values():
        start_frames.append(frame_to_time(faces['num_frames'][0]))
        end_frames.append(frame_to_time(faces['num_frames'][-1]))
        labels.append(faces['label_predicted'])
        planes.append(faces['plane_predicted'])

    metadata = [date.fromtimestamp(time.time())]*len(planes)

    dictionary_dataframe = {'0start_frames': start_frames,
                            '1end_frames': end_frames, '2label': labels, '3plane': planes, '4metadata': metadata}
    df = pd.DataFrame(data=dictionary_dataframe)
    df.to_csv('/home/yhoatsu/IdeaProjects/ML_News_Face_Recognition/relabeled_faces.csv')
