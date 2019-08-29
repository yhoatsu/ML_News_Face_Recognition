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
import datetime
#
import collections
#
import pandas as pd

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

# To save images images/frames or video, we save the base path where that script is located
name_folder_analysis = path_object.split('/')[-1].split('.')[0]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_ANALYSIS = os.path.join(BASE_DIR, name_folder_analysis)
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
with open(os.path.join(BASE_DIR_PICKLES, "metadata_faces_recognized.pickle"), 'rb') as f:
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

dict_relabel = {}
for label, faces_in_dict_final in dictionary_faces.items():
  
  if faces_in_dict_final['label_predicted']!='Unknown':

      list_pos = [index_label for index_label in range(len(faces_in_dict_final['labels'])) if faces_in_dict_final['labels'][index_label] == faces_in_dict_final['label_predicted']]

      last_value = 0
      jump = []
      for actual_pos in list_pos:
        if actual_pos - last_value > 10:
          jump = jump + [last_value, actual_pos]
        last_value = actual_pos
    
      if len(faces_in_dict_final['labels']) - last_value > 10:
          jump = jump + [last_value, len(faces_in_dict_final['labels'])]
    
      if len(jump) == 0:
        jump = [list_pos[-1]]
    
      cut_pos_init = 0   
      for pos_to_cut in range(len(jump)):
        if cut_pos_init == jump[pos_to_cut]:
          continue
      
        label_face = label+str(pos_to_cut)
        dict_relabel[label_face] = {}
        for dict_labels in ['position','labels','accuracy','num_frames','planes']:
          dict_relabel[label_face][dict_labels] = faces_in_dict_final[dict_labels][cut_pos_init:jump[pos_to_cut]]
      
        pred_face = predict_face(dict_relabel[label_face]['labels'])
        min_accuracy = min(dict_relabel[label_face]['accuracy'])
        pred_plane = predict_plane(dict_relabel[label_face]['planes'])  
      
        dict_relabel[label_face]['label_predicted'] = pred_face
        dict_relabel[label_face]['min_accuracy'] = min_accuracy
        dict_relabel[label_face]['plane_predicted'] = pred_plane

        if pred_face == 'Unknown':
            del dict_relabel[label_face]
         
        if pos_to_cut == len(jump)-1:
          dict_relabel[label_face+'0'] = {}
          for dict_labels in ['position','labels','accuracy','num_frames','planes']:
            dict_relabel[label_face+'0'][dict_labels] = faces_in_dict_final[dict_labels][jump[pos_to_cut]:]
        
          if len(dict_relabel[label_face+'0']['labels']) != 0:
            pred_face = predict_face(dict_relabel[label_face+'0']['labels'])
            min_accuracy = min(dict_relabel[label_face+'0']['accuracy'])
            pred_plane = predict_plane(dict_relabel[label_face+'0']['planes'])  

            dict_relabel[label_face+'0']['label_predicted']=pred_face
            dict_relabel[label_face+'0']['min_accuracy']=min_accuracy
            dict_relabel[label_face+'0']['plane_predicted']=pred_plane

          if pred_face == 'Unknown' or len(dict_relabel[label_face+'0']['labels']) == 0:
            del dict_relabel[label_face+'0']
      
        cut_pos_init = jump[pos_to_cut]

dictionary_faces = dict_relabel

start_frames = []
end_frames = []
labels = []
planes = []
accuracy = []

for faces_in_dict_final in dictionary_faces.values():
    start_frames.append(frame_to_time(faces_in_dict_final['num_frames'][0]))
    end_frames.append(frame_to_time(faces_in_dict_final['num_frames'][-1]))
    labels.append(faces_in_dict_final['label_predicted'])
    planes.append(faces_in_dict_final['plane_predicted'])
    accuracy.append(faces_in_dict_final['min_accuracy'])

# Save at a variable the termination of the object introduced by the user to analyze,
# to know if is a image or a video (same as .endswith(""))
type_object = path_object.split(".")[-1]

if type_object == 'mp4' or type_object == 'avi':

    video_in = cv2.VideoCapture(path_object)
    video_out = cv2.VideoWriter(os.path.join(BASE_DIR_ANALYSIS,'video_result_relabeled.avi'), cv2.VideoWriter_fourcc(*'XVID'),
                                min(25.0, video_in.get(5)), (int(video_in.get(3)), int(video_in.get(4))))
    num_frame = 0
    while video_in.isOpened():
        num_frame += 1
        ret, frame = video_in.read()
        for faces_in_dict_final in dictionary_faces.values():

            if num_frame in faces_in_dict_final['num_frames']:
                distance = faces_in_dict_final['min_accuracy']
                #if distance < 0.5:
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

    name_folder_analysis = name_folder_analysis = 'TD2_' + name_folder_analysis.replace('-','_').replace('_informatiu_nit','')

    metadata = 'Date of Analysis: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ', Video Analyzed: ' + name_folder_analysis + ', Characters to identify: Alberto_Rivera, Fernando_Giner, Isabel_Bonig, Joan_Ribo, Jose_Maria_Llanos, Maria_Jose_Catala, Maria_Oliver, Monica_Oltra, Pablo_Casado, Pablo_Iglesias, Pedro_Sanchez, Ruben_Martinez_Dalmau, Sandra_Gomez, Santiago_Abascal, Toni_Canto, Ximo_Puig, Jose_Gosalbez.' + ' Television_network: A punt, Video length: ' + frame_to_time(int(video_in.get(7))) 

    # When everything done, release the capture
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()

    dictionary_dataframe = {'0start_frames': start_frames,
                            '1end_frames': end_frames, '2label': labels, '3plane': planes, '4accuracy':accuracy, '5'+metadata: [' ']*len(start_frames)}
    df = pd.DataFrame(data=dictionary_dataframe)
    df.columns = ['Start_frames', 'End_frames', 'Label_predicted', 'Plane_predicted', 'Metric of accuracy', metadata]

    df.to_csv(os.path.join(BASE_DIR_ANALYSIS,'relabeled_faces.csv'), index=False)

    dict_parties = {'Pedro_Sanchez':'PSOE', 'Ximo_Puig':'PSOE', 'Sandra_Gomez':'PSOE', 
               'Pablo_Casado':'PP', 'Isabel_Bonig':'PP', 'Maria_Jose_Catala':'PP',
               'Pablo_Iglesias':'Podemos', 'Ruben_Martinez_Dalmau':'Podemos', 'Maria_Oliver':'Podemos',
               'Alberto_Rivera':'Ciudadanos', 'Toni_Canto':'Ciudadanos', 'Fernando_Giner':'Ciudadanos',
               'Monica_Oltra':'Compromis', 'Joan_Ribo':'Compromis',
               'Santiago_Abascal':'VOX', 'Jose_Maria_Llanos':'VOX', 'Jose_Gosalbez':'VOX'}

    data = df.iloc[:,:-1]

    for col_time in ['Start_frames', 'End_frames']:
        data[col_time] = pd.to_datetime(data[col_time], format="%H:%M:%S.%f")

    data['Time_s'] = data['End_frames'] - data['Start_frames']
    data['Time_s'] = data['Time_s']/np.timedelta64(1,'s')
    data['Parties'] = data.Label_predicted.map(dict_parties)
    
    df_time_appearance_pj = data.loc[:,['Label_predicted','Time_s']].groupby('Label_predicted').sum().reset_index()
    df_time_appearance_pj.columns = ['Character_name', 'Screen_time']
    df_time_appearance_pj[metadata] = [' ']*len(df_time_appearance_pj)

    df_time_appearance_parties = data.loc[:,['Parties','Time_s']].groupby('Parties').sum().reset_index()
    df_time_appearance_parties.columns = ['Party_name', 'Screen_time']
    df_time_appearance_parties[metadata] = [' ']*len(df_time_appearance_parties)

    df_time_appearance_parties_plane = data.loc[:,['Parties','Time_s','Plane_predicted']].groupby(['Parties','Plane_predicted']).sum().reset_index()
    df_time_appearance_parties_plane.columns = ['Party_name', 'Plane', 'Screen_time']
    df_time_appearance_parties_plane[metadata] = [' ']*len(df_time_appearance_parties_plane)

    df_time_appearance_pj_plane = data.loc[:,['Label_predicted','Time_s','Plane_predicted']].groupby(['Label_predicted','Plane_predicted']).sum().reset_index()
    df_time_appearance_pj_plane.columns = ['Character_name', 'Plane', 'Screen_time']
    df_time_appearance_pj_plane[metadata] = [' ']*len(df_time_appearance_pj_plane)

    df_time_appearance_pj.to_csv(os.path.join(BASE_DIR_ANALYSIS,'./time_appearance_leader.csv'), index=False)
    df_time_appearance_parties.to_csv(os.path.join(BASE_DIR_ANALYSIS,'time_appearance_parties.csv'), index=False)
    df_time_appearance_parties_plane.to_csv(os.path.join(BASE_DIR_ANALYSIS,'time_appearance_parties_plane.csv'),  index=False)
    df_time_appearance_pj_plane.to_csv(os.path.join(BASE_DIR_ANALYSIS,'time_appearance_leader_plane.csv'), index=False)

    df.to_csv(os.path.join(BASE_DIR_ANALYSIS,'Relabeled_faces.csv'), index=False)
