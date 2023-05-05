# Studienarbeit

## Dependencies

### Python environment

To set up a suitable virtual environment, execute the following commands:
```
sudo apt install python3-venv
python3 -m venv ipsp
# activate the environment
source ipsp/bin/activate
# install dependencies
pip install -r requirements.txt
# leave the environment
deactivate
```

### Data

The datasets are expected to be located in a folder named *data*, which is ignored by the version control system. 

## Getting started

A typical workflow might look as follows:
```
# starting from the repositories top level
source ipsp/bin/activate
cd notebooks
jupyter-lab
# select notebook and work until done ...
deactivate
```