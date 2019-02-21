FROM continuumio/miniconda3

RUN adduser --disabled-password cyclegan

WORKDIR /home/cyclegan

COPY env_requirements_linux.yml env_requirements_linux.yml 
RUN conda update -y conda
RUN conda update -y --all
RUN conda env create -f env_requirements_linux.yml
RUN conda config --add channels menpo 
RUN conda config --add channels anaconda
RUN conda config --add channels conda-forge
RUN conda config --add channels defaults
RUN conda install -y gunicorn
RUN conda install ca-certificates=2019.1.23 certifi=2018.11.29 libcxx=4 libedit=3.1.20181209 libffi=3.2.1 ncurses=6.1 openssl=1.1.1 pip=19.0.1 python=3.7.2 readline=7.0 setuptools=40.7.3 sqlite=3.26.0 tk=8.6.8 wheel=0.32.3 xz=5.2.4 zlib=1.2.11 click==7.0 flask==1.0.2 itsdangerous==1.1.0 jinja2==2.10 markupsafe==1.1.0 werkzeug==0.14.1 

COPY backend/application_server.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP application_server.py

RUN chown -R cyclegan:cyclegan ./
USER cyclegan

RUN echo "source activate cycleGAN" > ~/.bashrc
ENV PATH /opt/conda/envs/cycleGAN/bin:$PATH
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
