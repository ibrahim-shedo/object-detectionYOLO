# Real-Time Object Detection using YOLOv5

This project demonstrates how to perform real-time object detection using the YOLOv5 model with a webcam feed. The steps include installing dependencies, loading the YOLOv5 model, performing detection on static images, and integrating real-time detection with a webcam.

## Install and Import Dependencies

First, ensure that you have installed the required dependencies:

```bash
# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone the YOLOv5 repository from GitHub
!git clone https://github.com/ultralytics/yolov5 

# Navigate to the YOLOv5 folder
%cd yolov5

# Install required dependencies for YOLOv5
!pip install -r requirements.txt
Next, import the necessary Python libraries for the project:

python
Copy code
import torch  # Load the YOLO model and make detections
from matplotlib import pyplot as plt  # Visualize images
import numpy as np  # Handle array transformations
import cv2  # Access webcam and render feeds
from google.colab.patches import cv2_imshow  # Display images in Google Colab
Load the YOLOv5 Model
The YOLOv5 model can be loaded from the PyTorch Hub. Below is an example of how to load the pre-trained YOLOv5s model:

python
Copy code
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Display the model architecture
model
Make a Detection
To test the model, let's perform object detection on an example image:

python
Copy code
# Load an example image from a URL
img = 'https://ultralytics.com/images/zidane.jpg'

# Run the detection
results = model(img)

# Print detection results
results.print()

# Visualize the detection
%matplotlib inline
plt.imshow(np.squeeze(results.render()))
plt.show()
Real-Time Detection with Webcam
You can also use YOLOv5 to perform real-time object detection with a webcam feed. Here’s how:

Initialize the Webcam
python
Copy code
import cv2

cap = cv2.VideoCapture(0)  # Open the default camera

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

while True:
    ret, frame = cap.read()

    # Exit if no frame is captured
    if not ret:
        print("Failed to grab frame")
        break

    # Display the webcam feed
    cv2.imshow('Camera Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
Perform Real-Time Detection
Incorporate the YOLOv5 model into the webcam feed for real-time detection:

python
Copy code
cap = cv2.VideoCapture(0)  # Open the default camera

while cap.isOpened(): 
    ret, frame = cap.read()

    if not ret:
        break

    # Perform detection on the current frame
    results = model(frame)

    # Display the results
    cv2.imshow('YOLO', np.squeeze(results.render()))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
Display Webcam Feed without Detection
If you want to display the webcam feed without performing any detection:

python
Copy code
cv2.imshow('YOLO', frame)  # Display the webcam feed as is
Conclusion
This project showcases the usage of YOLOv5 for both static image detection and real-time object detection using a webcam. You can modify the code to fit your specific use case, including switching to other YOLOv5 models, adjusting detection thresholds, or integrating this with other machine learning pipelines.

markdown
Copy code

### Key Sections:
- **Dependencies**: Install necessary packages and set up the environment.
- **Loading the Model**: Show how to load the YOLOv5 model.
- **Static Image Detection**: Detect objects in a static image.
- **Real-Time Detection**: Integrate YOLOv5 with a webcam for real-time detection.
- **Simple Webcam Feed**: Show the webcam feed without performing detection.
