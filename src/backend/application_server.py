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
                  send_from_directory

from nn_modules import Generator
from torchvision.utils import save_image
import utils


ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg', 'ppm', 'pgm', 'tif'}

app = Flask(__name__)
#app.config.from_object('flask_configuration')
app.config['UPLOAD_FOLDER'] = Path.cwd() / 'uploads/'
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
app.config['RESPONSE_FOLDER'] = Path.cwd() / 'response/'
Path(app.config['RESPONSE_FOLDER']).mkdir(exist_ok=True)

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


def is_allowed_file(filename):
    """
    Given filename returns whether file is of allowed file type.
    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            app.logger.info('no file part')
            flash('No file part')
            return redirect(request.url)
        rcvd_file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if rcvd_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if rcvd_file and is_allowed_file(rcvd_file.filename):
            filename = secure_filename(rcvd_file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                              'tmp', filename)
            try:
                img = Image.open(rcvd_file)
                img = img.convert('RGB')
                prediction = model(transform(img).unsqueeze(0))
                filename_pred = 'prediction_{}.png'.format(
                    filename.rsplit('.', 1)[0])
                save_image(prediction,
                           os.path.join(app.config['RESPONSE_FOLDER'],
                                        filename_pred),
                           normalize=True)

            except:
                abort(404)
                abort(Response('Improper image type.'))


            '''
            # attempt to create image object from output torch.
            # Tensor, work in progress!
            data = prediction.data.numpy()
            new_img = transforms.ToPILImage(mode='RGB')(data)
            np_image = np.squeeze(prediction.data.numpy(), axis=0)
            np_image = np.transpose(np_image, (1, 2, 0))
            new_img = Image.fromarray(np_image, 'RGB')
            img_byte_arr = io.BytesIO()
            new_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            return send_file(io.BytesIO(img_byte_arr),
                     attachment_filename='prediction.png',
                     mimetype='image/png')'''

            return send_from_directory(app.config['RESPONSE_FOLDER'],
                                       filename_pred)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    load_model('gpu_model')
    load_transform_function()
