UPLOAD_FOLDER = Path.cwd() / 'uploads/'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
RESPONSE_FOLDER = Path.cwd() / 'response/'
Path(RESPONSE_FOLDER).mkdir(exist_ok=True)

# Set limit on file size of uploaded files. 50 * 1024 * 1024 is 50 MB.
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
UPLOAD_FOLDER = UPLOAD_FOLDER
RESPONSE_FOLDER = RESPONSE_FOLDER