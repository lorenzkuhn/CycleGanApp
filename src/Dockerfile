FROM continuumio/miniconda3

RUN adduser --disabled-password cyclegan

WORKDIR /home/cyclegan

COPY backend/templates/index.html templates/index.html
# TODO Lorenz: Copy entire backend/static folder instead of individual files
COPY backend/static/styles/style.css static/styles/style.css
COPY backend/static/favicon.ico static/favicon.ico
COPY environment.yml environment.yml
COPY learning/gpu_model gpu_model
COPY learning/nn_modules.py nn_modules.py
COPY learning/utils.py utils.py
RUN conda update -y conda
RUN conda update -y --all
RUN conda env create -f environment.yml

COPY backend/application_server.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP application_server.py

RUN chown -R cyclegan:cyclegan ./
USER cyclegan

EXPOSE 80
ENTRYPOINT [ "/bin/bash", "-c", "source activate onenv && ./boot.sh" ] 
