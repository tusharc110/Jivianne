import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("modelTRANSFER.keras")

def predict_image(img):
    img = img.resize((224, 224))
    
    img_array = image.img_to_array(img) / 255.0  
    
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  
    
    class_labels = {0: "Class 0", 1: "Class 1"}  
    return f"Prediction: {class_labels[predicted_class]} ({predictions[0][predicted_class]:.2f} confidence)"

interface = gr.Interface(
    fn=predict_image,  
    inputs=gr.Image(type="pil"),  
    outputs="text",  
    title="Transfer Model Prediction",  
    description="Upload an image to test the Transfer model."
)
interface.launch()