import tensorflow as tf

model_path = './CNN_trained_model.keras'
model = tf.keras.models.load_model(model_path)

import cv2
import numpy as np

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

'''
For static image classification :
    
img2 = cv2.imread('glass269.jpg')
resized_frame = cv2.resize(img2, (300, 300))  # Resize to model's input size
normalized_frame = resized_frame / 255.0      # Normalize pixel values
input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(input_frame)
predicted_class = labels[np.argmax(predictions)]
print(predicted_class)
'''

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (300, 300))  # Resize to model's input size, same as during CNN training process!
    normalized_frame = resized_frame / 255.0  
    input_frame = np.expand_dims(normalized_frame, axis=0)  

    predictions = model.predict(input_frame)
    predicted_class = labels[np.argmax(predictions)]

    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

