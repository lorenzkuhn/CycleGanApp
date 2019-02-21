#!/bin/sh
source activate cycleGAN
exec gunicorn -b :5000 --access-logfile - --error-logfile - application_server:app
