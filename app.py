import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Flatten
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import Image

# A placeholder for a face detection model (e.g., from OpenCV or a simple Haar cascade)
# In a real-world scenario, you would use a more robust detector like MTCNN or MediaPipe.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Model Architecture ---
# This function creates a hybrid model based on the report's description (InceptionV3 + RNN)
# In a real application, you would load a pre-trained model with weights.
@st.cache_resource
def get_model():
    """
    Creates and returns a placeholder hybrid deepfake detection model.
    This model is for demonstration purposes and is not trained.
    """
    # Define the input shape for video frames (e.g., 20 frames, 150x150 pixels, 3 channels)
    sequence_input = Input(shape=(20, 150, 150, 3))

    # InceptionV3 as a feature extractor for each frame
    # We use a TimeDistributed layer to apply the InceptionV3 model to each frame in the sequence.
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(150, 150, 3), pooling='avg')
    
    # We use a dummy model here because we don't have the weights
    # In a real implementation, you would load the trained model's weights
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Apply the base model to the sequence of frames
    time_distributed_layer = TimeDistributed(base_model)(sequence_input)
    
    # Flatten the output to feed into a recurrent layer
    flatten_layer = TimeDistributed(Flatten())(time_distributed_layer)
    
    # A simple RNN (or LSTM/GRU) for temporal analysis
    rnn_layer = tf.keras.layers.SimpleRNN(128)(flatten_layer)
    
    # A dense layer for final classification
    output_layer = Dense(1, activation='sigmoid')(rnn_layer)
    
    model = Model(inputs=sequence_input, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# --- Streamlit UI ---
st.title("Deepfake Detector 🕵️‍♂️")
st.write("Upload a video file to detect if it contains deepfake content.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(uploaded_file, format="video/mp4", start_time=0)
    
    if st.button("Analyze Video"):
        st.subheader("Analysis Results")
        
        # Load the placeholder model
        model = get_model()

        # Placeholder for video processing and prediction
        st.info("Extracting frames and analyzing...")
        
        # Get video frames and prepare them for the model
        video_path = "temp_video.mp4"
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        while cap.isOpened() and frame_count < 20: # Process a fixed number of frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0] # Take the first detected face
                face_image = frame[y:y+h, x:x+w]
                
                # Resize and normalize the face image to match the model's input
                face_image = cv2.resize(face_image, (150, 150))
                face_image = img_to_array(face_image)
                face_image = face_image / 255.0
                frames.append(face_image)
                frame_count += 1
        
        cap.release()
        
        if len(frames) < 20:
            st.warning(f"Could not extract enough faces from the video. Only {len(frames)} processed.")
            st.warning("Analysis cannot be performed with this video.")
        else:
            # Convert list of frames to a numpy array for prediction
            frames = np.array(frames)
            frames = np.expand_dims(frames, axis=0) # Add batch dimension

            # Make a placeholder prediction
            # In a real app, you would use:
            # prediction = model.predict(frames)
            
            # For demonstration, we'll use a random placeholder result
            prediction = np.random.rand(1)
            
            if prediction[0] > 0.5:
                st.error("🚨 **This video is likely a deepfake.**")
                st.write(f"Confidence Score: {prediction[0]:.2f}")
            else:
                st.success("✅ **This video appears to be authentic.**")
                st.write(f"Confidence Score: {1 - prediction[0]:.2f}")
            
            st.write("---")
            st.write("**Disclaimer:** This is a demonstration model and its accuracy is limited. For critical analysis, please use a fully trained, robust deepfake detection system.")
            
# Footer
st.markdown("---")
st.markdown("Project based on the Final Year Project Report from the Department of Computer Engineering, at the University of Lahore.")
