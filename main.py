import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from keras import regularizers
import tensorflow as tf


emotion_index = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
# json_file = open('emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

emotion_model = Sequential()
# #1st cnn layer
emotion_model= tf.keras.models.Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
emotion_model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# #2nd cnn layer 
emotion_model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# #3rd cnn layer
emotion_model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# #4th cnn layer 
emotion_model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# #5th cnn layer
emotion_model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
# #fully connected 1st layer
emotion_model.add(Flatten()) 
emotion_model.add(Dense(256,activation = 'relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.25))
# #fully connected 2nd layer    
emotion_model.add(Dense(512,activation = 'relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(7, activation='softmax'))

# load weights into new model
emotion_model.load_weights("best_model.h5")
print("Executed..")

# start the webcam feed
video = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = video.read()
    frame = cv2.resize(frame, (1000, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    number_of_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in number_of_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 0, 255), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_image = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_image)
        max_index = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_index[max_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key== ord('q'):
        break

video.release()
cv2.destroyAllWindows()