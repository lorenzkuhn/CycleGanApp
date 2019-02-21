# CycleGanApp
## Deployment using Docker Containers
We used digitalocean to deploy our app in a docker container.
1. Run this command from within the 'src' directory:  docker build -t application_server:latest .
2. Start the docker container with the following command: docker run --name application_server -d -p 80:2375 --rm application_server:latest

### Debugging Docker Container
Given the container is currently running, follow these steps:
1. Get id of container using command 'docker ps'
2. docker exec -it `<docker id>` /bin/bash

### Using the Conda Environment

To create the environment file:
1. Activate the environment, i.e. `source activate myenv`.
2. Run `conda env export > env_requirements.yml`

To create the environment from an environment file:
1. If you'd like, change the environment name in the first line of env_requirements.yml
2. Run: `conda env create -f env_requirements.yml`
3. Enjoy your new environment. 

To update an existing environment if the environment file changes:
1. `source deactivate`
2. `conda env update -f env_requirements.yml`
3. `source activate env_name`

### Running the Flask app

In the folder of the flask app:
```
$ export FLASK_APP=hello.py
$ flask run
```
