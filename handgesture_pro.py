import cv2                           #computer vision for realtime imagine processing
import numpy as np                   #store the numpy array of data points
import os                            #interaction between user and operating system
from matplotlib import pyplot as plt #plotting the graph
import time                          #sleep between each frame
import mediapipe as mp               #extract keypoints saving as frames

mp_holistic = mp.solutions.holistic # pre-trained machine learning model that can detect and track multiple human body landmarks such as facial landmarks, hands, and body posture.
mp_drawing = mp.solutions.drawing_utils # user-friendly visualization of the detected landmarks.

#OpenCV uses BGR color as a default color space to display images
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 #Make prediction ie model takes images as non writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #COLOR COVERSION RGB 2 BGR
    return image, results

mp_drawing.draw_landmarks #function to draw landmarks on frames

def draw_styled_landmarks(image, results):

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
    # Draw pose line connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )


cap = cv2.VideoCapture(0) #accessing th web camera

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#min_detection_confidence is the minimum confidence score required for a detection
#min_tracking_confidence is the minimum confidence score required for a tracking
    while cap.isOpened(): #loop through the feed

        # Read feed from camera
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show screen to the user
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully from camera feed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release() #release the web camera
    cv2.destroyAllWindows() #close down the window

cap.release() #release the web camera just in case camera doesn't close
cv2.destroyAllWindows() #close down the window

len(results.left_hand_landmarks.landmark) # number of landmark on lefthand

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) #flatten is used to covert to one array from list of array
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

extract_keypoints(results).shape #shape of extracted point is 258. 33*4 + 21*3 + 21*3

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['done','hello','thankyou','yes'])

# fifty videos worth of data
no_sequences = 50

# Videos are going to be 50 frames in length
sequence_length = 50

#collect 50 frames per video and 50 vidoes per action.

for action in actions: 
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences ie each videos
        for sequence in range(1,no_sequences+1):
            # Loop through video length ie sequence length
            for frame_num in range(1,sequence_length+1):

                # Read feed through web cam
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # wait logic
                if frame_num == 1: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image) # Show to screen
                    cv2.waitKey(50) #wait logic between each video
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(50)
                    
                # Export keypoints from each frames
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully from web camera
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()


from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)} #labeling each actions

#sequences represent x label
#labels represent y label
sequences, labels = [], []
for action in actions:
    for sequence in range(1,no_sequences+1):
        window = []
        for frame_num in range(1,sequence_length+1):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res) #adding frame to the window
        sequences.append(window) #appending all the videos to squences label
        labels.append(label_map[action]) #labelling all the vidoes based on action


np.array(sequences).shape #(number of videos,frames,datapoints)

np.array(labels).shape #(number of videos,labels)

X = np.array(sequences)

X.shape

y = to_categorical(labels).astype(int)

y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=100, test_size=0.2)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

X_val.shape

y_val.shape

# 8. Build and Train LSTM Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,258)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

actions.shape[0] #output labels

opt = Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[tb_callback,early_stop])


#tensorboard --logdir=. --bind_all
model.summary()


# plot training and validation accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


res = model.predict(X_test)

res #probability given by softmax

actions[np.argmax(res[1])]

actions[np.argmax(y_test[1])]

model.save('Modelppt0.h5')

from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, f1_score, precision_score, recall_score, classification_report
import seaborn as sns


yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()

yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)

# create the confusion matrix
cm = confusion_matrix(ytrue, yhat)
print(cm)


sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

f1 = f1_score(ytrue, yhat, average='macro')
precision = precision_score(ytrue, yhat, average='macro')
recall = recall_score(ytrue, yhat, average='macro')

print('F1 score: {:.3f}'.format(f1))
print('Precision: {:.3f}'.format(precision))
print('Recall: {:.3f}'.format(recall))

print(classification_report(ytrue, yhat))

accuracy_score(ytrue, yhat)

# 1. New detection variables
sequence = []
#sentence = []
#threshold = 0.7

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence = sequence[:50]
        
        if len(sequence) == 50:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #print(actions[np.argmax(res)])

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


action2 = plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

prediction2 = actions[np.argmax(res)]

prediction2



