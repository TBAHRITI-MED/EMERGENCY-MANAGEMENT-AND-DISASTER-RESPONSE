import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model_emergencyNet.h5')

# Define class labels
class_labels = {
    0: 'collapsed_building',
    1: 'Fire',
    2: 'Flood',
    3: 'Normal/None',
    4: 'Traffic Incident'
}

# Function to preprocess images
def preprocess_image(image):
    # Convert to float32
    image = image.astype(np.float32)
    # Normalize
    image = (image / 127.5) - 1
    return image

# Function to classify image
def classify_image(image):
    # Resize image to match model input shape
    resized_image = cv2.resize(image, (240, 240))
    # Preprocess image
    preprocessed_image = preprocess_image(resized_image)
    # Expand dimensions to match model input shape
    input_image = np.expand_dims(preprocessed_image, axis=0)
    # Perform inference
    predictions = model.predict(input_image)
    # Get class probabilities
    probabilities = predictions[0]
    # Get class labels and probabilities
    results = [(class_labels[i], probabilities[i]) for i in range(len(class_labels))]
    # Sort results by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Function to display results
def display_results(image, results):
    # Loop through results and display probabilities
    for i, (label, prob) in enumerate(results):
        text = '{}: {:.2f}%'.format(label, prob * 100)
        cv2.putText(image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Display image
    cv2.imshow('Classification', image)

# Main function
def main():
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Classify frame
        results = classify_image(frame)

        # Display results
        display_results(frame, results)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()