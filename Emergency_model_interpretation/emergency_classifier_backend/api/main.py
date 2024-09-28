import os
import time
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model, get_model_inference_function, get_channel_configuration_bands
from ml4floods.data.worldfloods import dataset
from ml4floods.visualization import plot_utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('images', 'inputs')
app.config['OUTPUT_FOLDER'] = os.path.join('images', 'outputs')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def save_file(file, path):
    content = file.read()
    try:
        with open(path, 'wb') as f:
            f.write(content)
    except Exception as e:
        return {"message": "There was an error uploading the file"}
    finally:
        file.close()

def load_model(experiment_name, device):
    folder_name_model_weights = os.path.join("checkpoints", experiment_name)
    config_fp = os.path.join(folder_name_model_weights, "config.json")

    config = get_default_config(config_fp)
    config["model_params"]["max_tile_size"] = 128

    model_folder = os.path.dirname(folder_name_model_weights)
    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True
    model = get_model(config["model_params"], experiment_name)

    model = model.eval()
    model = model.to(device)

    return model, config

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    experiment_name = request.form.get('experiment_name', 'WF2_unet')
    if experiment_name not in ['WF2_unet', 'WFV1_scnn20', 'WFV1_unet']:
        return jsonify({'error': 'Invalid experiment name'}), 400

    filename = f'{str(time.time()).replace(".", "_")}_{secure_filename(file.filename)}'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = save_file(file, file_path)

    if result is not None:
        return jsonify(result), 500

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
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    plt.figure(figsize=(10, 10))
    plt.imshow(prediction.cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.savefig(output_path)
    plt.close()

    url = f'/images/outputs/{output_filename}'
    return render_template('result.html', url=url)

@app.route('/images/outputs/<filename>')
def send_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
