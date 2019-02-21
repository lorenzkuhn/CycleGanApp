#!/bin/sh
source activate cycleGAN
# log to stdout in debug mode
exec gunicorn --workers=3 --timeout 120 -b 0.0.0.0:2375 --access-logfile - --error-logfile - --log-level=debug application_server:app
# log to access.log and error.log in non-debug mode
# exec gunicorn --workers=3 --timeout 120 -b 0.0.0.0:2375 --access-logfile ./access.log --error-logfile ./error.log --log-level=debug application_server:app


