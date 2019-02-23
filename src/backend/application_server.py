
"""
Module defining our flask application server.
"""

import os
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from flask import Flask, flash, request, redirect, url_for, session,\
                  render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from gan import Generator
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

UPLOAD_FOLDER = Path.cwd() / 'uploads/'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
RESPONSE_FOLDER = Path.cwd() / 'response/'
Path(RESPONSE_FOLDER).mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'ppm', 'pgm', 'tif'])

app = Flask(__name__)
app.secret_key = b'MBWUdbxX;>]vrTL'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESPONSE_FOLDER'] = RESPONSE_FOLDER
model = Generator(9)
transform = None

def init_inference(model_path):
    """
    Loads model from file and initialises image transformation.
    :param model_path:
    :return:
    """
    global model
    global transform
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    image_size = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def allowed_file(filename):
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

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            app.logger.info('no file part')
            flash('No file part')
            return redirect(request.url)
        rcvd_file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if rcvd_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # TODO Lorenz: Cover case where file type is not allowed
        # TODO Lorenz: Enforce max file size limitation
        if rcvd_file and allowed_file(rcvd_file.filename):
            filename = secure_filename(rcvd_file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                              'tmp', filename)
            app.logger.info("received file {}".format(uploaded_file_path))
            img = Image.open(rcvd_file)
            img = img.convert('RGB')
            app.logger.info('created image {}'.format(img))
            prediction = model(transform(img).unsqueeze(0))
            filename_pred = 'prediction_{}.png'.format(filename.split('.')[0])
            save_image(prediction, 
                       os.path.join(app.config['RESPONSE_FOLDER'], filename_pred), 
                       normalize=True)
            return send_from_directory(app.config['RESPONSE_FOLDER'], filename_pred)


            app.logger.info("predicted file {}".format(filename_pred))
            save_image(prediction, os.path.join(app.config['RESPONSE_FOLDER'],
                                                filename_pred), normalize=True)
            return send_from_directory(app.config['RESPONSE_FOLDER'],
                                       filename_pred)
            '''
            # attempt to create image object from output torch.Tensor, work in progress!
            data = prediction.data.numpy()            
            new_img = transforms.ToPILImage(mode='RGB')(data)
            np_image = np.squeeze(prediction.data.numpy(), axis=0)
            np_image = np.transpose(np_image, (1, 2, 0))
            for j in range(3):
                min_value = np.min(np_image[:, :, j])
                max_value = np.max(np_image[:, :, j])
                if min_value == max_value:
                    print('channel: ' + str(j) + ' ; min: ' + str(min_value) +
                        ' ; max: ' + str(max_value))
                    np_image[:, :, j] = .5
                np_image[:, :, j] = (np_image[:, :, j] - min_value) /\
                    (max_value - min_value)
            new_img = Image.fromarray(np_image, 'RGB')
            img_byte_arr = io.BytesIO()
            new_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            return send_file(io.BytesIO(img_byte_arr),
                     attachment_filename='prediction.png',
                     mimetype='image/png')'''

    return render_template('index.html')


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    init_inference('gpu_model')
