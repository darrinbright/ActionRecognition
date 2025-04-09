from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import os
import numpy as np

model = load_model('model.h5')

def process_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Walking.")
    else:
        print("Running.")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scale_factor = 0.5 

    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (new_width, new_height))

    frame_count = 0
    predictions = []
    sliding_window_size = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame_for_model = cv2.resize(frame, (150, 150))  
        img_array = img_to_array(resized_frame_for_model)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)[0]
        predictions.append(prediction)

        if len(predictions) > sliding_window_size:
            predictions.pop(0)

        average_prediction = np.mean(predictions)
        class_label = 'Walking' if average_prediction > 0.5 else 'Running'

        resized_frame_for_display = cv2.resize(frame, (new_width, new_height))  
        cv2.putText(resized_frame_for_display, f'{class_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 

        out.write(resized_frame_for_display)

        cv2.imshow('Video', resized_frame_for_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

source = 'walk.mp4'
file_extension = os.path.splitext(source)[1].lower()

if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
    process_image(source)
elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
    process_video(source)
else:
    print("Unsupported file type.")