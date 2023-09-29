# A Comparison of Reduced-Order Models for Wing Buffet Predictions

![Alt Text](output/LSTM/single/VAE/param_study/VAE_LSTM_single_reconstruction.gif)
## Abstract

The primary aim of this study is to train and evaluate a set of Reduced-Order Models (ROMs) for
predicting upper-wing pressure distributions on a civil aircraft configuration. Using wind tunnel
data recorded for the Airbus XRF-1 research configuration in the European Transonic Windtun-
nel (ETW), the ROMs integrate dimensionality reduction and time-evolution components. For
dimensionality reduction, both Singular Value Decomposition (SVD) and a convolutional Vari-
ational Autoencoder (CNN-VAE) neural network are employed to reduce/encode Instationary
Pressure Sensitive Paint (IPSP) data from a transonic buffet flow condition. Fully-Connected
(FC) and Long Short-Term Memory (LSTM) neural networks are then applied to predict the
evolution of the latent representation, which is subsequently reconstructed/decoded back to its
original high-dimensional state.

SVD and CNN-VAE are trained on data from five flow conditions and tested on two unseen
conditions. When comparing the power spectra of the reconstructed and experimental data, SVD
demonstrates marginally better performance. Subsequently, FC and LSTM models are applied
for the forward evolution of the reduced/latent representation for one flow condition, resulting in
four evaluated ROMs. Among them, the CNN-VAE-LSTM model excels by accurately capturing
buffet dynamics, encompassing both transient features and steady-state oscillations. SVD-based
models tend to struggle with transient buffet behavior, as they predominantly learn steady-state
dynamics. Additionally, the CNN-VAE-LSTM model was employed for an end-to-end training
approach. The results suggest that it faces difficulties maintaining dynamics in predictions,
demanding further investigation and optimization.

## BibTex Citation

```
@misc{schreiber2023,
  author       = {Anton Schreiber},
  title        = {A Comparison of Reduced-Order Models for Wing Buffet Predictions},
  month        = nov,
  year         = 2023,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {???}
}
```


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
- Load your data with .pt format into the *data* folder with:
    1. a $c_p$-snapshot file containing different flow conditions, that should be stored in a dictionary. The *keys* represent the flow condition name and the corresponding *values* represent the pressure data in form of PyTorch Tensors with (height, width, number of snapshots)
    2. a coordinate or mesh file, that could also be stored in  similar dictionary format. The *keys* also represent the flow condition, while the *values* are tuples containing coordinate meshes for the height and the width dimension

(**Note for all following steps that you probably need to change path names in the scripts and adjust parameters in the *app/utils/config.py* file according to your specific data**)

- Interpolate your data and limit the number of timesteps with the *interpolate_coords* and *make_data_subset* functions in *app/preprocessing.py*. This is optional but can significantly reduce required computational ressources.
- In *app/preprocessing.py*, use *svd_preprocesing* and *autoencoder_preprocessing* to create datasets for the training of SVD and CNN-VAE. 
- In the next step, you can train the dimensionality reduction techniques:
    - **SVD**: Simply run the *SVD_train.ipynb* file in the *notebooks* directory.
    - **CNN-VAE**: The training of a CNN-VAE neural network is more complex. Head to *autoencoder* directory in *app* and open *train_VAE.py*. This file is used to start a parameter study for different bottleneck layer sizes of the CNN-VAE models. All training parameters are specified in the *config.py* file, adjust them to your needs (this also applies for all other neueral network trainings)
- To monitor the training, use the corresponding *<>_optim.ipynb* notebooks in the *notebooks* directory. To apply both techniques to test data, use the corresponding *<>_test.ipynb* notebooks. Adjust parameters and repeat the training (optional).
- Similarly, FC and LSTM neural networks can be trained to auto-regressively predict the time-series in the reduced/ latent space. In *app*, there are *FC* and *LSTM* directories where you can find the model classes and the training scripts. There, you can set the dimensionality reduction technique to pair the model with and start a training pipelines. In the frame of the course project, they were trained on a single flow condition using the *single_flow_cond_preprocessing* function in the *preprocessing.py*. You can also copy and adjust the scripts to train with multiple flow conditions, the pre-processing scripts are already available. The only difference is that you should consider a 3rd dataset when dealing with multiple flow conditions to end up with a training, a validation and a test dataset, which is not sensible for a single flow condition.
- In the last step, use corresponding *<>_optim.ipynb* and *<>_test.ipynb* notebooks to monitor the training and test the models on unseen data. 

## Extensibility

This repository can be extended by adding new dimensionality reduction and time-evolution techniques in the same modular structure.