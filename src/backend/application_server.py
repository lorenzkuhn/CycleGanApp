"""
Module defining our flask application server.
"""

import os
import logging
from pathlib import Path
from PIL import Image
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from flask import Flask, flash, request, redirect, render_template,\
                  send_from_directory, abort, send_file
import io
from nn_modules import Generator
from torchvision.utils import save_image
import utils
import _thread
import time
import random

random.seed()
ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg', 'ppm', 'pgm', 'tif'}
TARGET_IMAGE_SIZE = (3, 256, 256)
app = Flask(__name__)
#app.config.from_object('flask_configuration')
app.config['UPLOAD_FOLDER'] = '/persistentlogs/uploads'
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Set limit on file size of uploaded files. 50 * 1024 * 1024 is 50 MB.
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

model = Generator(n_residual_blocks=9, use_dropout=False)
transform = None


def load_model(model_path):
    """
    Loads model from file and initialises image transformation.
    :param model_path:
    :return:
    """
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))


def load_transform_function():

    global transform
    image_size = (256, 256)
    transform = utils.get_transform(image_size)

def store_image(filename, uploaded_file):
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


def start_save_image_thread(uploaded_file, filename):
    try:
        filename = 'upload_{}_{}.{}'.format(time.strftime("%Y%m%d-%H%M%S"), 
                                            random.randint(0,100000), 
                                            extract_file_ending(filename))
        _thread.start_new_thread( store_image, (filename, uploaded_file, ) )
    except:
        app.logger.info("Failed to store file {}".format())
    

def extract_file_ending(filename):
    return filename.rsplit('.', 1)[1].lower()

def is_allowed_file(filename):
    """
    Given filename returns whether file is of allowed file type.
    :param filename:
    :return:
    """
    return '.' in filename and \
           extract_file_ending(filename) in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handles upload of image.
    :return:
    """
    if request.method == 'GET':
        return render_template('index.html')
    else:   # POST request
        if ('file' not in request.files or
                ('file' in request.files and request.files['file'] is None)):
            abort(404)
            return

        rcvd_file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if rcvd_file.filename == '':
            abort(404)
            return

        if not rcvd_file or not is_allowed_file(rcvd_file.filename):
            abort(404)
            return

        start_save_image_thread(rcvd_file, rcvd_file.filename)

        try:
            img = Image.open(rcvd_file)
            img = img.convert('RGB')
            prediction = model(transform(img).unsqueeze(0))
            prediction = prediction.reshape(TARGET_IMAGE_SIZE)
            return serve_pil_image(prediction)

        except:
            abort(404)
            return


def serve_pil_image(prediction):
    """
    Returns inferred image to user without writing image to disk.
    :param pil_img:
    :return:
    """
    prediction = prediction + torch.ones(list(TARGET_IMAGE_SIZE))
    prediction = prediction / (2 * torch.ones(list(TARGET_IMAGE_SIZE)))
    new_img = transforms.ToPILImage(mode='RGB')(prediction)
    img_byte_arr = io.BytesIO()
    new_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return send_file(io.BytesIO(img_byte_arr),
        attachment_filename='prediction.png',
        mimetype='image/png')

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    load_model('gpu_model')
    load_transform_function()
