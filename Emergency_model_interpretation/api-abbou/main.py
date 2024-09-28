import os
import torch
import time
from fastapi import FastAPI,Query,UploadFile,File,Request
from fastapi.staticfiles import StaticFiles
from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data.worldfloods import dataset

app = FastAPI()

app.mount("/public", StaticFiles(directory="images"), name="static")

IMAGES_DIR = os.path.join('.','images')
INPUTS_DIR = os.path.join(IMAGES_DIR, 'inputs')
OUTPUTS_DIR = os.path.join(IMAGES_DIR, 'outputs')

def save_file(file : UploadFile, path : str):
    
    content = file.file.read()

    try:
        with open(path, 'wb') as f:
            f.write(content)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

def load_model(experiment_name, device):

    folder_name_model_weights = os.path.join(".","checkpoints",experiment_name)
    config_fp = os.path.join(folder_name_model_weights, "config.json")

    ### Configs
    config = get_default_config(config_fp)
    config["model_params"]["max_tile_size"] = 128

    ### Load the model
    model_folder = os.path.dirname(folder_name_model_weights)
    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)

    model = model.eval()
    model = model.to(device)

    return model,config

@app.post("/")
async def root(
    request: Request,
    file : UploadFile = File(),
    experiment_name : str = Query(default="WF2_unet", regex='(WF2_unet|WFV1_scnn20|WFV1_unet)'),
):
    
    filename = f'{str(time.time()).replace('.','_')}_{file.filename}'
    file_path = os.path.join(INPUTS_DIR, filename)
    result = save_file(file, file_path)

    if result is not None:
        return result
    
    ### Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model,config = load_model(experiment_name, device)

    ### Get inference function
    inference_function = get_model_inference_function(model, config,apply_normalization=True)

    ### Load the file
    window = None
    channel_configuration = config.model_params.hyperparameters.channel_configuration
    channels = get_channel_configuration_bands(channel_configuration)
    torch_inputs, transform = dataset.load_input(file_path, window=window, channels=channels)

    # Make predictions
    outputs = inference_function(torch_inputs.unsqueeze(0))[0] # (num_classes, h, w)
    prediction = torch.argmax(outputs, dim=0).long() # (h, w)

    # Post processing
    mask_invalid = torch.all(torch_inputs == 0, dim=0)
    prediction+=1
    prediction[mask_invalid] = 0

    # Save prediction
    file_path = os.path.join(OUTPUTS_DIR, os.path.splitext(filename)[0] + os.path.extsep + 'pt')
    torch.save(prediction, file_path)

    # Response
    base_url = str(request.url).split('?')[0][:-1]
    filename = os.path.splitext(filename)[0] + os.path.extsep + 'pt'
    url = os.path.join(base_url,'public','outputs',filename)

    return { 
        "url" : url
    }
