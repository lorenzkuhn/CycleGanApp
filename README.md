# CycleGanApp
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
