# Automatic Translation

This repository contains automatic translation tools for the Flores200 languages using the NLLB model from Meta AI (https://github.com/facebookresearch/fairseq/tree/nllb). 

Features include:

- Translation to any of the Flores200 languages
- Detection of the source language from Flores200 languages
- Detection of toxicity in text from Flores200 languages

The tools are available to use as a module or as an API.

## Usage

To learn how to use these tools, check out the `/examples` folder.

There is also a frontend application built with Streamlit that consumes the API. To learn more, check out the following repository: https://github.com/rosasalberto/automatic_translation_frontend

## Installation Guide

1. Install CUDA 11.6
   - Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
   - Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
2. Install Torch for CUDA 11.6
   - Check https://pytorch.org/get-started/locally/
   ```console
   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```
3. Verify that Torch with Cuda has been correctly installed
   ```console
   >>> import torch
   >>> torch.cuda.is_available() # should return True
   >>> torch.cuda.device_count() # should return > 0
4. Clone this repo
    ```console
    git clone https://github.com/rosasalberto/automatic_translation_server
    ```
5. Install pipenv
    ```console
    pip install pipenv
    ```
6. Install the needed dependencies in a virtual environment and activate it
    ```console
    pipenv install
    pipenv shell
    ```
7. Download Language Detection (LID) model from the provided link: https://tinyurl.com/nllblid218e
8. Configure the server by modifying the `config.py` file:
   1. Modify `translation_langs` to include the languages you want to be able to translate, using the Flores200 language codes.
   2. Modify `lid_path` to the full path of the LID model.
   3. Modify `path_toxicity_data` to the full path to the toxicity vocab files.

## Run Server

#### Start server:
```console
uvicorn server:app --reload
```

#### API Swagger
http://127.0.0.1:8000/docs

## TODO

2) Dockerize
3) Deployment on cloud