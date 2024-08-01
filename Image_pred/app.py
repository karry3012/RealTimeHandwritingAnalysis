# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# # Load the trained model
# model = load_model('model_cnn.h5')

# # Function to preprocess the image
# def preprocess_image(image):
#     # Convert the image to grayscale
#     image = image.convert('L')
#     # Resize the image to 28x28 pixels
#     image = image.resize((28, 28))
#     # Convert the image to a numpy array
#     image_array = np.array(image)
#     # Normalize the image array
#     image_array = image_array / 255.0
#     # Reshape the image for CNN input (add batch and channel dimensions)
#     image_array = image_array.reshape(1, 28, 28, 1)
#     return image_array

# # Function to make predictions
# def predict(image):
#     processed_image = preprocess_image(image)
#     predictions = model.predict(processed_image)
#     predicted_digit = np.argmax(predictions)
#     return predicted_digit

# # Streamlit app
# def main():
#     st.title("MNIST Digit Recognizer")
    
#     # Description
#     st.write("Upload an image of a digit (0-9) and the model will predict the digit.")
    
#     # Image uploader
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         # Make prediction
#         prediction = predict(image)
        
#         # Display the prediction
#         st.write(f"Prediction: {prediction}")

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np


# model = load_model('model_cnn.h5')


# def preprocess_image(image):
#     image = image.convert('L')
#     image = image.resize((28, 28))
#     image_array = np.array(image)
#     image_array = image_array / 255.0
#     image_array = image_array.reshape(1, 28, 28, 1)
#     return image_array


# def predict(image):
#     processed_image = preprocess_image(image)
#     predictions = model.predict(processed_image)
#     predicted_digit = np.argmax(predictions)
#     return predicted_digit

# # Function to retrain the model
# def retrain_model(image, correct_digit):
#     processed_image = preprocess_image(image)
#     x_train = processed_image
#     y_train = np.array([correct_digit])
    
#     # Recompile the model
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     # Fit the model with the new data
#     model.fit(x_train, y_train, epochs=5)
#     model.save('model_cnn.h5')


# def main():
    
#     st.title('Digit Recognizer with Active Learning Model :sunglasses:')
#     st.write("Upload an image of a digit (0-9) and the model will predict the digit")
    
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         prediction = predict(image)
#         st.write(f"Prediction: {prediction}")
        
#         # Ask user if they're satisfied with the prediction
#         satisfied = st.radio("Are you satisfied with this prediction?", ('Yes', 'No'))
        
#         if satisfied == 'No':
#             correct_digit = st.number_input("Please enter the correct digit:", min_value=0, max_value=9, step=1)
#             if st.button("Submit correct digit"):
#                 retrain_model(image, correct_digit)
#                 st.write("Model retrained successfully. Thank you for your feedback!")
                
#                 # Make a new prediction with the retrained model
#                 new_prediction = predict(image)
#                 st.write(f"New prediction after retraining: {new_prediction}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time


st.set_page_config(page_title="Digit Recognizer", page_icon=":1234:", layout="wide")


model = load_model('model_cnn.h5')

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_digit = np.argmax(predictions)
    return predicted_digit


def retrain_model(image, correct_digit):
    processed_image = preprocess_image(image)
    x_train = processed_image
    y_train = np.array([correct_digit])
    
   
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    model.fit(x_train, y_train, epochs=5)
    model.save('model_cnn.h5')




def main():
    st.sidebar.title("Digit Recognizer with Active Learning :sunglasses:")
    st.sidebar.write("Upload an image of a digit (0-9) and the model will predict the digit")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            time.sleep(1)
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
            prediction = predict(image)
            st.write(f"**Prediction:** {prediction}")
        
            
            satisfied = st.radio("Are you satisfied with this prediction?", ('Yes', 'No'))
        
            if satisfied == 'No':
                correct_digit = st.number_input("Please enter the correct digit:", min_value=0, max_value=9, step=1)
                if st.button("Submit correct digit"):
                    with st.spinner('Retraining model...'):
                        retrain_model(image, correct_digit)
                        st.success("Model retrained successfully. Thank you for your feedback!")
                        
                        
                        new_prediction = predict(image)
                        st.write(f"**New prediction after retraining:** {new_prediction}")

    st.sidebar.markdown("---")
    st.sidebar.write("Powered by Streamlit")


st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #20232a;
        color: white;
    }
    .stButton>button {
        background-color: #61dafb;
        color: black;
    }
    .stRadio>div>div>label {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()



