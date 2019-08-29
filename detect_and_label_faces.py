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
from time import time
import datetime
#
import collections
#
import pandas as pd
import shutil

dlib.DLIB_USE_CUDA = True

# To check if the params are well introduced by the user
if len(sys.argv) != 8:
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
use_save_data = sys.argv[7]

# To save images images/frames or video, we save the base path where that script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

name_folder_analysis = path_object.split('/')[-1].split('.')[0]

BASE_DIR_ANALYSIS = os.path.join(BASE_DIR, name_folder_analysis)
BASE_DIR_VIDEOS = os.path.join(BASE_DIR_ANALYSIS, 'video_temp')
BASE_DIR_PICKLES = os.path.join(BASE_DIR_ANALYSIS, 'pickles')

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
def my_knn(face_descriptor_to_predict, coef=0, training_set=x_train, labels_training=y_labels,
           neighborhoods=3, ord_norm=2, threshold=threshold_user):
    # Calculates the L^n norm for each pair of values training_set[q]-face_descriptor_to_predict
    # By defect calculates L2 norm (euclidean distance)
    distances = np.linalg.norm(training_set - face_descriptor_to_predict, axis=1, ord=ord_norm) + coef
    # To order the array of distances and save the arguments or locations of the values of distances
    locations_lower_distances = np.argsort(distances)

    # If the min distance between the introduced face descriptor and the faces at the training set
    # aren't to close (defined by the threshold) then stop here and return as Unknown the face under analysis
    lower_distance = distances[locations_lower_distances[0]]
    if lower_distance < threshold:

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
    return prediction, lower_distance


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


def predict_plane(labels):
    count_rep_elements = collections.Counter(labels)
    del count_rep_elements['Plano general']
    dict_count_rep_elements = count_rep_elements.most_common(1)
    if len(dict_count_rep_elements) == 0:
        most_freq_element = 'Plano general'
    else:
        most_freq_element = dict_count_rep_elements[0][0]
    return most_freq_element


def frame_to_time(frame):
    time_raw = frame/25
    time_milli_secs = int((time_raw - int(time_raw))*100)
    time_secs_acum = int(time_raw)
    time_min = int(time_secs_acum/60)
    time_hours = int(time_min/60)
    time_min = time_min - time_hours*60
    time_secs = time_secs_acum - time_min*60
    return str(time_hours) + ':' + str(time_min) + ':' + str(time_secs) + '.' + str(time_milli_secs)


# Save at a variable the termination of the object introduced by the user to analyze,
# to know if is a image or a video (same as .endswith(""))
type_object = path_object.split(".")[-1]

if type_object == 'mp4' or type_object == 'avi':

    video_in = cv2.VideoCapture(path_object)

    total_frames = int(video_in.get(7))
    num_frames_percentagje = round(total_frames/100)

    num_frame_init = 0
      
    if use_save_data == 'True':
        for variable in ['num_frame', 'time_faces', 'accuracy', 'labels_faces', 'labels_planes', 'dictionary_faces', 'dictionary_faces_temp']:
            with open(os.path.join(BASE_DIR_PICKLES, variable+'.pickle'), 'rb') as file:
                exec(variable+ '= pickle.load(file)')

        label_video = len(os.listdir(BASE_DIR_VIDEOS))
        if num_frame < total_frames:
            video_out = cv2.VideoWriter(os.path.join(BASE_DIR_VIDEOS, 'video_result_all_faces_'+str(label_video)+'.avi'), cv2.VideoWriter_fourcc(*'XVID'),
                                min(25.0, video_in.get(5)), (int(video_in.get(3)), int(video_in.get(4))))
    else:
        #shutil.rmtree(BASE_DIR_VIDEOS)
        num_frame = 0
        time_faces = []
        labels_faces = []
        labels_planes = []
        accuracy = []
        #
        dictionary_faces = {}
        dictionary_faces_temp = {}
        dictionary_faces_temp['init'] = \
        {'position': [{'left': -1, 'right': -1, 'top': -1, 'bottom': -1}],
        'labels': [],
        'accuracy': [],
        'num_frames': []}
        os.makedirs(BASE_DIR_VIDEOS)
        os.makedirs(BASE_DIR_PICKLES)
        label_video = len(os.listdir(BASE_DIR_VIDEOS))
        video_out = cv2.VideoWriter(os.path.join(BASE_DIR_VIDEOS, 'video_result_all_faces_'+str(label_video)+'.avi'), cv2.VideoWriter_fourcc(*'XVID'),
                                min(25.0, video_in.get(5)), (int(video_in.get(3)), int(video_in.get(4))))

    while video_in.isOpened():

        if num_frame_init < num_frame and use_save_data == 'True':
            ret, frame = video_in.read()
            num_frame_init += 1
            continue
        elif num_frame >= total_frames:
            break

        num_frame += 1
        num_frame_init += 1

        ret, frame = video_in.read()
        #frames.append(frame)
        faces = face_detector(frame)
        for face in faces:
            # The detector return different objects depend if we use cnn or hog
            if face_rec_model == "cnn":
                # Here we get the landmarks that defines the face.
                shape = shape_pred(frame, face.rect)
                top, right, bottom, left = face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()
            else:
                shape = shape_pred(frame, face)
                top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()

            width_face = right - left
            length_face = bottom - top

            width_ratio = width_face/video_in.get(3)
            length_ratio = length_face/video_in.get(4)

            coef_adjustment = (1-(width_ratio*length_ratio)**(1/2))*0.06

            center_face = {'axis_x': left+width_face/2, 'axis_y': top+length_face/2}
            area_face = width_face*length_face
            axis_x_ratio = video_in.get(3)/center_face['axis_x']
            axis_y_ratio = video_in.get(4)/center_face['axis_y']

            type_plane = 'Plano general'
            #if 1.5 <= axis_x_ratio <= 2.5:
            if 1.5 <= axis_y_ratio <= 2.5:
                if 0.8 <= width_ratio <= 1 and 0.55 <= length_ratio <= 1:
                    type_plane = 'Primerísimo primer plano'
            if 2 <= axis_y_ratio <= 3:
                if 0.13 <= width_ratio <= 0.2 and 0.2 <= length_ratio <= 0.55:
                    type_plane = 'Primer plano'
            if 3 <= axis_y_ratio <= 4:
                if 0.1 <= width_ratio <= 0.16 and 0.13 <= length_ratio <= 0.24:
                    type_plane = 'Plano central'
                if 0.07 <= width_ratio <= 0.13 and 0.11 <= length_ratio <= 0.2:
                    type_plane = 'Plano medio'
            if 5.5 <= axis_y_ratio <= 6.5:
                if 0.16 <= width_ratio <= 0.8 and 0.11 <= length_ratio <= 0.2:
                    type_plane = 'Plano americano'
            if 4 <= axis_y_ratio <= 5:
                if 0 <= width_ratio <= 0.1 and 0.06 <= length_ratio <= 0.13:
                    type_plane = 'Plano medio largo'
            if 3.5 <= axis_y_ratio <= 4.5:
                if 0.07 <= width_ratio <= 0.13 and 0 <= length_ratio <= 0.11:
                    type_plane = 'Plano entero'

            # Computes the 128D array that define the face and that'll be used to predict the label of the face.
            face_descriptor = np.array(face_rec.compute_face_descriptor(frame, shape, resample_times))
            #
            label_predicted, distance = my_knn(face_descriptor, coef=coef_adjustment, neighborhoods=1)
            #
            if label_predicted == 'Unknown':
                #time_faces.append(frame_to_time(num_frame))
                #labels_faces.append(label_predicted)
                #labels_planes.append(type_plane)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.putText(frame, label_predicted+str(round(distance,2)), (left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                time_faces.append(frame_to_time(num_frame))
                labels_faces.append(label_predicted)
                labels_planes.append(type_plane)
                accuracy.append(distance)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
                cv2.putText(frame, label_predicted+str(round(distance,2)), (left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            face_found = 0
            there_are_faces = 0
            for faces_in_dict in dictionary_faces_temp.values():

                there_are_faces = 1
                last_position = faces_in_dict['position'][-1]

                if last_position['left'] <= center_face['axis_x'] <= last_position['right'] and \
                        last_position['top'] <= center_face['axis_y'] <= last_position['bottom']:

                    width_last_face = last_position['right'] - last_position['left']
                    length_last_face = last_position['bottom'] - last_position['top']
                    center_last_face = {'axis_x': last_position['left']+width_face/2,
                                        'axis_y': last_position['top']+length_face/2}
                    #area_last_face = width_last_face*length_last_face
                    # and abs(area_last_face-area_face) <= video_in.get(3)*video_in.get(4)*0.01

                    if abs(center_last_face['axis_x']-center_face['axis_x']) <= video_in.get(3)*0.25 and \
                            abs(center_last_face['axis_y']-center_face['axis_y']) <= video_in.get(4)*0.25:

                        faces_in_dict['position'].append({'left': left, 'right': right, 'top': top, 'bottom': bottom})
                        faces_in_dict['labels'].append(label_predicted)
                        faces_in_dict['accuracy'].append(distance)
                        faces_in_dict['planes'].append(type_plane)
                        faces_in_dict['num_frames'].append(num_frame)

                    face_found = 1

            if face_found == 0 and there_are_faces == 1:
                dictionary_faces_temp['face_' + str(time())] = \
                    {'position': [{'left': left, 'right': right, 'top': top, 'bottom': bottom}],
                     'labels': [label_predicted], 'accuracy': [distance], 'planes': [type_plane], 'num_frames': [num_frame]}

            try:
                del dictionary_faces_temp['init']
            except KeyError:
                pass

            key_to_delete = []
            for key, faces_in_dict in dictionary_faces_temp.items():
                if num_frame-faces_in_dict['num_frames'][-1] >= 10:
                    key_to_delete.append(key)
                    pred_face = predict_face(faces_in_dict['labels'])
                    min_accuracy = min(faces_in_dict['accuracy'])
                    pred_plane = predict_plane(faces_in_dict['planes'])
                    #if pred_face != 'Unknown':
                    face_name = 'face_' + str(time())
                    dictionary_faces[face_name] = faces_in_dict
                    dictionary_faces[face_name]['label_predicted'] = pred_face
                    dictionary_faces[face_name]['min_accuracy'] = min_accuracy
                    dictionary_faces[face_name]['plane_predicted'] = pred_plane

            for keys in key_to_delete:
                del dictionary_faces_temp[keys]

            if len(dictionary_faces_temp) == 0:
                dictionary_faces_temp['init'] = \
                    {'position': [{'left': -1, 'right': -1, 'top': -1, 'bottom': -1}],
                    'labels': [],
                    'accuracy': [],
                    'num_frames': []}
                        
        video_out.write(frame)

        if verbose == "yes":
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #percentagje_analized = (num_frame/total_frames)*100

        if num_frame%num_frames_percentagje == 0:
            percentagje = int(num_frame/num_frames_percentagje)
            print("Porcentaje del vídeo analizado: ", percentagje, "%")
            
            if percentagje%25 == 0 and percentagje != 100:
                for element_to_save in ['num_frame', 'time_faces', 'accuracy', 'labels_faces', 'labels_planes', 'dictionary_faces', 'dictionary_faces_temp']:
                    with open(os.path.join(BASE_DIR_PICKLES, element_to_save+'.pickle'), 'wb') as file:
                        exec('pickle.dump('+element_to_save+', file)')
                #sys.exit()

        #if abs(int(percentagje_analized)-round(percentagje_analized,1)) < 0.0009:
        #    print("Porcentaje del vídeo analizado: ", int(percentagje_analized), "%")

        
        #for element_to_save in ['num_frame', 'time_faces', 'labels_faces', 'labels_planes', 'dictionary_faces', 'dictionary_faces_temp']:
         #   with open(os.path.join(BASE_DIR_PICKLES, element_to_save+'.pickle'), 'wb') as file:
          #      exec('pickle.dump('+element_to_save+', file)')

        if num_frame == int(video_in.get(7))+1:
            for element_to_save in ['num_frame', 'time_faces', 'accuracy', 'labels_faces', 'labels_planes', 'dictionary_faces', 'dictionary_faces_temp']:
                with open(os.path.join(BASE_DIR_PICKLES, element_to_save+'.pickle'), 'wb') as file:
                    exec('pickle.dump('+element_to_save+', file)')
            break

    video_out.release()
    video_out_final = cv2.VideoWriter(os.path.join(BASE_DIR_ANALYSIS, 'video_result_all_faces.avi'), cv2.VideoWriter_fourcc(*'XVID'),
                                min(25.0, video_in.get(5)), (int(video_in.get(3)), int(video_in.get(4))))
    
    metadata = 'Date of Analysis: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ', Video Analyzed: ' + name_folder_analysis + ', Characters to identify: Alberto_Rivera, Fernando_Giner, Isabel_Bonig, Joan_Ribo, Jose_Maria_Llanos, Maria_Jose_Catala, Maria_Oliver, Monica_Oltra, Pablo_Casado, Pablo_Iglesias, Pedro_Sanchez, Ruben_Martinez_Dalmau, Sandra_Gomez, Santiago_Abascal, Toni_Canto, Ximo_Puig, Jose_Gosalbez.' + ' Television_network: A punt, Video length:' + frame_to_time(int(video_in.get(7))) 

    video_in.release()

    for video_labels in range(len(os.listdir(BASE_DIR_VIDEOS))):

        video_in = cv2.VideoCapture(os.path.join(BASE_DIR_VIDEOS, 'video_result_all_faces_'+str(video_labels)+'.avi'))
        iter=0
        while video_in.isOpened():
            iter+=1
            ret, frame = video_in.read()
            video_out_final.write(frame)
            if iter >= int(video_in.get(7))-1:
                break
        video_in.release()
    video_out_final.release()

    #metadata = ['Date of Analysis: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  'Video Analyzed: '+name_folder_analysis, 'Characters to identify: Alberto_Rivera, Fernando_Giner, Isabel_Bonig, Joan_Ribo, Jose_Maria_Llanos, Maria_Jose_Catala, Maria_Oliver, Monica_Oltra, Pablo_Casado, Pablo_Iglesias, Pedro_Sanchez, Ruben_Martinez_Dalmau, Sandra_Gomez, Santiago_Abascal, Toni_Canto, Ximo_Puig']
    #metadata = metadata + [' ']*(len(time_faces)-3)

    dictionary_dataframe = {'Time': time_faces, 'Label_predicted': labels_faces, 'Plane_predicted': labels_planes, 'Metric of accuracy': accuracy, metadata: [' ']*len(time_faces)}
    df = pd.DataFrame(data=dictionary_dataframe)
    
    df.to_csv(os.path.join(BASE_DIR_ANALYSIS,'recognized_faces.csv'), index=False)

    with open(os.path.join(BASE_DIR_PICKLES, 'metadata_faces_recognized.pickle'), 'wb') as f:
        pickle.dump(dictionary_faces, f)

    shutil.rmtree(BASE_DIR_VIDEOS)
    cv2.destroyAllWindows()

