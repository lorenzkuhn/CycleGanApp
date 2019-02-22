FROM continuumio/miniconda3

RUN adduser --disabled-password cyclegan

WORKDIR /home/cyclegan

COPY backend/templates/index.html templates/index.html
COPY backend/static/styles/style.css static/styles/style.css
COPY backend/static/favicon.ico static/favicon.ico
COPY ../onenv_requirements.yml environment.yml
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
