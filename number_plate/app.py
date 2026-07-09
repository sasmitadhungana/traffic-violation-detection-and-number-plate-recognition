from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Page Title
st.title("Automated Traffic Violation Monitoring")
st.write("Nepali License Plate Detection using YOLO11")

# Load Model
model = YOLO(r"F:\best.pt")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.image(image, caption="Original Image")

    # Convert image to numpy array
    img_array = np.array(image)

    # Run YOLO detection
    results = model.predict(
        source=img_array,
        conf=0.5
    )

    # Draw bounding boxes
    result_img = results[0].plot()

    # Show result
    st.image(
        result_img,
        caption="Detected License Plates"
    )

    # Number of detections
    num_plates = len(results[0].boxes)

    st.success(
        f"Detected {num_plates} license plate(s)"
    )

    # Display confidence scores
    if num_plates > 0:
        st.subheader("Detection Details")

        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            st.write(
                f"Plate {i+1}: Confidence = {conf:.2f}"
            )