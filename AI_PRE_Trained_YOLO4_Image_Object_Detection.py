import cv2  # Import the OpenCV library
import numpy as np  # Import the NumPy library for numerical computations

# Load YOLO model
net = cv2.dnn.readNet("/Users/praveen18kumar/Downloads/yolov4.weights", "/Users/praveen18kumar/Downloads/yolov4.cfg")  # Load pre-trained YOLO weights and configuration
classes = []  # Initialize an empty list to store class names
with open("/Users/praveen18kumar/Downloads/coco.names", "r") as f:  # Open the file containing class names
    classes = [line.strip() for line in f.readlines()]  # Read class names and strip newline characters
layer_names = net.getLayerNames()  # Get names of all layers in the network
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Get output layer names

# Loading image
img = cv2.imread("/Users/praveen18kumar/Downloads/Leh-Ladakh-bike-trip-from-Srinagar-1.jpg")  # Load image from file
img = cv2.resize(img, None, fx=0.4, fy=0.4)  # Resize image for faster processing
height, width, channels = img.shape  # Get image dimensions

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Preprocess image as required by YOLO model
net.setInput(blob)  # Set input to neural network
outs = net.forward(output_layers)  # Forward pass to get output predictions

# Initialize list to store indices of objects that have been drawn
drawn_objects = []

# Showing information on the screen
for out in outs:
    for detection in out:
        scores = detection[5:]  # Get confidence scores for all classes
        class_id = np.argmax(scores)  # Get class ID with highest confidence
        confidence = scores[class_id]  # Get confidence score for the detected class
        if confidence > 0.5:  # Check if confidence score is above threshold
            # Object detected
            center_x = int(detection[0] * width)  # Calculate center x-coordinate of bounding box
            center_y = int(detection[1] * height)  # Calculate center y-coordinate of bounding box
            w = int(detection[2] * width)  # Calculate width of bounding box
            h = int(detection[3] * height)  # Calculate height of bounding box

            # Rectangle coordinates
            x = int(center_x - w / 2)  # Calculate x-coordinate of top-left corner of bounding box
            y = int(center_y - h / 2)  # Calculate y-coordinate of top-left corner of bounding box

            # Check if this object has already been drawn
            if class_id not in drawn_objects:
                # Draw rectangle and text
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Draw bounding box
                cv2.putText(img, classes[class_id], (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)  # Add class label

                # Add index of drawn object to the list
                drawn_objects.append(class_id)

# Display image with detected objects
cv2.imshow("Image", img)  # Show the image with annotations
cv2.waitKey(0)  # Wait for any key press
cv2.destroyAllWindows()  # Close all OpenCV windows
