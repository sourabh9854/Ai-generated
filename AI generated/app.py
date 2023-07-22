import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = tf.keras.models.load_model('my_final_model2.hdf5')
    return model
  
def get_model():
    return load_model()

def import_and_predict(image_data, model):
    size = (256, 256)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def main():
    st.write("""
             # AI GENERATED IMAGE OR NOT
"""
             )
    model = get_model()
    class_names = ['AI', 'Real']
    
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.write("Class:", predicted_class)

        print("This image most likely belongs to {} with a {:.2f} percent confidence."
              .format(predicted_class, confidence))

if __name__ == '__main__':
    main()
