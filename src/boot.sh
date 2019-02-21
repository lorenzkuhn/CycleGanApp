#!/bin/sh
source activate cycleGAN
exec gunicorn --workers=2 -b 0.0.0.0:2375 --access-logfile - --error-logfile - application_server:app


