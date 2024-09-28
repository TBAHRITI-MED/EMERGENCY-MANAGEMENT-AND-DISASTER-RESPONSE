import os
import time
import torch
import logging
from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model as ld
from tensorflow.keras.layers import SpatialDropout2D, LeakyReLU
from PIL import Image
import numpy as np
from .source import preprocess_input_image, batch_predict, conv_float_int, combine_image, burn_area
import cv2
import base64
from io import BytesIO
from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model, get_model_inference_function, get_channel_configuration_bands
from ml4floods.data.worldfloods import dataset
import pprint
import json
from django.conf import settings
import torch
from torchvision.utils import save_image
from django.http import HttpResponse
from django.views.decorators import gzip
import threading
from django.http import StreamingHttpResponse , HttpResponseServerError
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
from matplotlib import pyplot as plt




logger = logging.getLogger(__name__)

# Define directories for input and output images
IMAGES_DIR = os.path.join('.', 'images')
INPUTS_DIR = './inputs'  # Update to your actual input directory
OUTPUTS_DIR = os.path.join(settings.MEDIA_ROOT, 'outputs')  # Ensure this matches your media settings
# Ensure directories exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# Ensure directories exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def home(request):
    return render(request, 'landing/index.html')

def predict_emergency(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            # Load the EmergencyNet model from the classifier directory
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_emergencyNet.h5')
            custom_objects = {'SpatialDropout2D': SpatialDropout2D, 'LeakyReLU': LeakyReLU}
            model = ld(model_path, custom_objects=custom_objects)

            # Process the uploaded image and make predictions using the EmergencyNet model
            image = request.FILES['image']
            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (240, 240))
            img = img / 255.0  # Normalize

            # Reshape the image to match model input shape (batch_size, height, width, channels)
            img = np.expand_dims(img, axis=0)

            # Make predictions
            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            classes = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
            result = {'class': classes[class_index], 'confidence': float(prediction[0][class_index])}

            return render(request, 'predict_result.html', {'result': result})
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            return render(request, 'predict_result.html', {'error': str(e)})
    else:
        return render(request, 'upload_image.html')

def predict_fire_damage_map(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            # Load the wildfire detection model
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_model.h5')
            custom_objects = {'SpatialDropout2D': SpatialDropout2D, 'LeakyReLU': LeakyReLU}
            model = ld(model_path, custom_objects=custom_objects)

            # Process the uploaded image
            image = request.FILES['image']
            uploaded_image = Image.open(image)
            input_image_array = np.array(uploaded_image)
            original_width, original_height, pix_num = input_image_array.shape

            # Preprocess the image
            new_image_array, row_num, col_num = preprocess_input_image(input_image_array)

            # Make predictions
            preds = batch_predict(new_image_array, model)
            output_pred = conv_float_int(combine_image(preds, row_num, col_num, original_width, original_height, remove_ghost=True)[:, :, 0])
            
            # Create output mask
            preds_t = (preds > 0.25).astype(np.uint8)
            output_mask = conv_float_int(combine_image(preds_t, row_num, col_num, original_width, original_height, remove_ghost=False)[:, :, 0])

            # Ensure the arrays are uint8
            output_pred = np.uint8(output_pred)
            output_mask = np.uint8(output_mask)

            # Calculate burn area and CO2 emissions
            forest_type = request.POST.get('forest_type', 'Tropical Forest')
            resolution = float(request.POST.get('resolution', '10'))
            area, biomass_burnt, equal_days = burn_area(output_mask=output_mask, resolution=resolution, forest_type=forest_type)

            # Convert images to base64
            def image_to_base64(image_array):
                image = Image.fromarray(image_array)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')

            original_image_base64 = image_to_base64(input_image_array)
            predicted_image_base64 = image_to_base64(output_pred)
            mask_image_base64 = image_to_base64(output_mask)

            # Prepare context for rendering
            context = {
                'original_image': original_image_base64,
                'predicted_image': predicted_image_base64,
                'mask_image': mask_image_base64,
                'area': area / 1e6,
                'biomass_burnt': biomass_burnt / 1e6,
                'equal_days': equal_days
            }

            return render(request, 'result.html', context)
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return render(request, 'upload_image.html')


def predict_flood_damage(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            experiment_name = request.POST.get('experiment_name', 'WF2_unet')

            if experiment_name not in ['WF2_unet', 'WFV1_scnn20', 'WFV1_unet']:
                return JsonResponse({'error': 'Invalid experiment name'}, status=400)

            filename = f'{str(time.time()).replace(".", "_")}_{file.name}'
            file_path = os.path.join(INPUTS_DIR, filename)
            result = save_file(file, file_path)

            if result is not None:
                return JsonResponse(result, status=500)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, config = load_model(experiment_name, device)
            inference_function = get_model_inference_function(model, config, apply_normalization=True)
            window = None
            channels = get_channel_configuration_bands(config['model_params']['hyperparameters']['channel_configuration'])
            torch_inputs, transform = dataset.load_input(file_path, window=window, channels=channels)

            outputs = inference_function(torch_inputs.unsqueeze(0))[0]
            prediction = torch.argmax(outputs, dim=0).long()
            mask_invalid = torch.all(torch_inputs == 0, dim=0)
            prediction += 1
            prediction[mask_invalid] = 0

            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(OUTPUTS_DIR, output_filename)

            # Use Matplotlib to save the prediction as an image
            plt.figure(figsize=(10, 10))
            plt.imshow(prediction.cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.savefig(output_path)
            plt.close()

            base_url = request.build_absolute_uri('/')
            url = os.path.join(base_url, 'media', 'outputs', output_filename)

            # Render a template with the URL
            return render(request, 'flood_result.html', {'url': url})

        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return render(request, 'upload_image_flood.html')

def save_file(file, path):
    try:
        with open(path, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)
        return None
    except Exception as e:
        logger.error(f"Error uploading the file: {str(e)}")
        return {"message": "There was an error uploading the file"}

def load_model(experiment_name, device):
    folder_name_model_weights = os.path.join(".", "checkpoints", experiment_name)
    config_fp = os.path.join(folder_name_model_weights, "config.json")

    if not os.path.exists(config_fp):
        raise FileNotFoundError(f"Config file not found: {config_fp}")

    config = get_default_config(config_fp)
    logger.info("Loaded config:")
    logger.info(pprint.pformat(config))

    # Ensure correct access to nested configuration keys
    try:
        max_tile_size = config['model_params']['hyperparameters']['max_tile_size']
    except KeyError as e:
        raise KeyError(f"Config file is missing {e} key")

    logger.info(f"Max tile size: {max_tile_size}")

    model = get_model(config['model_params'], experiment_name)
    model = model.eval()
    model = model.to(device)

    return model, config

# Ensure get_default_config correctly returns a dictionary-like object
def get_default_config(config_fp):
    with open(config_fp, 'r') as f:
        config = json.load(f)
    return config

# Define the get_model function
def get_model(model_params, experiment_name):
    # Mock function, replace with actual model loading logic
    model = torch.nn.Module()  # Placeholder for actual model loading
    return model

# Mock dataset and related functions
class dataset:
    @staticmethod
    def load_input(file_path, window, channels):
        # Mock function, replace with actual data loading logic
        data = torch.zeros((len(channels), 256, 256))  # Placeholder tensor
        return data, None

# Mock get_channel_configuration_bands function
def get_channel_configuration_bands(channel_configuration):
    # Mock function, replace with actual logic to get channel bands
    return list(range(13))  # Example: 13 channels

# Mock get_model_inference_function function
def get_model_inference_function(model, config, apply_normalization):
    # Mock function, replace with actual inference logic
    def inference_fn(inputs):
        return torch.zeros((1, 2, inputs.shape[2], inputs.shape[3]))  # Placeholder output
    return inference_fn

def tensor_to_image(request, filename):
    # Assuming the tensor is stored in a .pt file
    tensor = torch.load(os.path.join('path_to_tensors', filename))

    # Assuming tensor is an image tensor
    image_path = 'path_to_save_image/image.png'
    save_image(tensor, image_path)

    # Read the image and send it as a response
    with open(image_path, 'rb') as f:
        return HttpResponse(f.read(), content_type="image/png")

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_emergencyNet.h5')
custom_objects = {'SpatialDropout2D': SpatialDropout2D, 'LeakyReLU': LeakyReLU}
model = ld(model_path, custom_objects=custom_objects)

class_labels = {
    0: 'collapsed_building',
    1: 'Fire',
    2: 'Flood',
    3: 'Normal/None',
    4: 'Traffic Incident'
}

# Function to preprocess images
def preprocess_image(image):
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    return image

# Function to classify image
def classify_image(image):
    resized_image = cv2.resize(image, (240, 240))
    preprocessed_image = preprocess_image(resized_image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    probabilities = predictions[0]
    results = [(class_labels[i], probabilities[i]) for i in range(len(class_labels))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def generate_frames():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not start camera.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture image")
                break

            results = classify_image(frame)
            for i, (label, prob) in enumerate(results):
                text = f'{label}: {prob:.2f}%'
                cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
    except Exception as e:
        logger.error("Error in video feed: %s", str(e))
        raise RuntimeError("Error generating frames.")

def real_time_detection(request):
    return render(request, 'real_time_detection.html')

@gzip.gzip_page
def video_feed(request):
    try:
        return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    except RuntimeError as e:
        logger.error("Error in video feed: %s", str(e))
        return HttpResponseServerError(str(e))
