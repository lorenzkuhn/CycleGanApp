FROM continuumio/miniconda3

RUN adduser --disabled-password cyclegan

WORKDIR /home/cyclegan

COPY env_requirements_linux.yml env_requirements_linux.yml 
RUN conda update -y conda
RUN conda update -y --all
RUN conda env create -f env_requirements_linux.yml
RUN conda install -y gunicorn
RUN conda install Flask

COPY backend/application_server.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP application_server.py

RUN chown -R cyclegan:cyclegan ./
USER cyclegan

RUN echo "source activate cycleGAN" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
