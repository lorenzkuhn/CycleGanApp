import os, logging
from flask import Flask, flash, request, redirect, url_for, session, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
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
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = b'MBWUdbxX;>]vrTL'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESPONSE_FOLDER'] = RESPONSE_FOLDER
model = Generator(9)
transform = None

def init_inference(model_path):
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
    #validation_data = torchvision.datasets.ImageFolder(root=VALIDATION_DATA_PATH, transform=transform)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    app.logger.info('upload_folder = {}'.format(app.config['UPLOAD_FOLDER']))
    app.logger.info('response_folder = {}'.format(app.config['RESPONSE_FOLDER']))
    app.logger.info(request)
    app.logger.info(request.files)
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
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            app.logger.info("received file {}".format(uploaded_file_path))
            rcvd_file.save(uploaded_file_path)
            prediction = model(transform(Image.open(uploaded_file_path)).unsqueeze(0))
            filename_pred = 'prediction_{}.png'.format(filename.split('.')[0])

            app.logger.info("predicted file {}".format(filename_pred))
            save_image(prediction, os.path.join(
                app.config['RESPONSE_FOLDER'], filename_pred) , normalize=True)

            return send_from_directory(app.config['RESPONSE_FOLDER'],
                               filename_pred)

    return render_template('index.html')

# Leaving this in for now to display how to do redirects:
# return redirect(url_for('uploaded_file',
            #                        filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    init_inference('awsm_model')
