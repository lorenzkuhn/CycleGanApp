#!/bin/bash
# Use the following to pipe log to stdout in debug mode
#exec gunicorn --workers=5 --timeout 120 -b 0.0.0.0:8080 --access-logfile - --error-logfile - --log-level=debug application_server:app

# Use this to write log to access.log and error.log (redirects all std output to error.log)
exec gunicorn --workers=5 --timeout 120 -b 0.0.0.0:8080 --access-logfile ./access.log --error-logfile ./error.log --log-level=debug --capture-output application_server:app
