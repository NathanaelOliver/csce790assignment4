import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set the paths
model_path = "model.h5"
test_images_dir = "/test_data/test_images/"
output_csv_file = "inference_results.csv"

# Load the TensorFlow model
model = tf.keras.models.load_model(model_path)

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = "converted_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Perform inference on test images
results = []
for i, img_name in enumerate(os.listdir(test_images_dir)):
    img_path = os.path.join(test_images_dir, img_name)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    start_time = time.time()
    interpreter.invoke()
    latency = (time.time() - start_time) * 1000  # in milliseconds
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    
    results.append({
        "id": i + 1,
        "image_id": img_name,
        "label": predicted_class,
        "latency": latency
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_file, index=False)
