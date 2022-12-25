# Automatic Translation

This repository contains automatic translation tools for the Flores200 languages using the NLLB model from Meta AI (https://github.com/facebookresearch/fairseq/tree/nllb). 

This API allows you to translate text to 200 languages automatically, detect the source language of a text from 200 languages, and get toxicity in texts.

## Endpoints

<<<<<<< HEAD
![Swagger Image](./media/swagger.PNG)
=======
![Swagger Image](media/swagger.PNG)
>>>>>>> 31a5265a24b8f8fc8f7101f3842989e8dbb84c52

### `/langs`

Get all the available translation languages.

**Method**: `GET`

#### Response

A list with all the available languages using FLORES-200 code.

### `/translate`

Automatic translation using NLLB model from Meta AI. Translate input_text in langs_out languages. Available languages: `[all languages listed in /langs endpoint]`.

**Method**: `POST`

#### Body

| Field       | Type   | Required | Description                                             |
|-------------|--------|----------|---------------------------------------------------------|
| input_text  | string | Yes      | The text to be translated.                              |
| langs_out   | string | No       | The languages to translate the input text to (comma separated). If not specified, it will translate to all languages. |

#### Response

The result of the translation service, containing translated text in all the languages specified.

### `/toxicity`

Get toxicity in texts without specifying the source language.

**Method**: `POST`

#### Body

| Field       | Type   | Required | Description                                             |
|-------------|--------|----------|---------------------------------------------------------|
| input_text  | string | Yes      | The text to be checked for toxicity.                    |

#### Response

The result of the toxicity detection service, containing toxic words in input_text.

### `/detect`

Detect source language of a text from 200 languages.

**Method**: `POST`

#### Body

| Field       | Type   | Required | Description                                             |
|-------------|--------|----------|---------------------------------------------------------|
| input_text  | string | Yes      | The text to detect the language of.                     |

#### Response

The result of the language detection service, containing the detected language of the input text.

## Usage

To learn how to use these tools, check out the `/examples` folder. There is also a frontend application built with Streamlit that consumes this API. To learn more, check out the following repository: https://github.com/rosasalberto/automatic_translation_frontend.

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
7. Download Language Detection (LID) model from the provided link: https://tinyurl.com/nllblid218e and add id to the '/weights' folder
8. Download the NLLB model 'pytorch_model.bin' from https://drive.google.com/drive/folders/1PejK0WhWsY3RJ9c3zEowNzUpctoEfyTY?usp=share_link and add it to the '/hub/models--facebook--nllb-200-distilled-600M/snapshots/368f64e5d5437e922548864bc115edcaa97aed60' folder
9. Configure the server by modifying the `config.py` file:
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

#### Test streamlit frontend application
- Set up application from https://github.com/rosasalberto/automatic_translation_frontend

## Build your docker Image
To build a Docker image, you need to have Docker installed on your machine. If you don't have it already, you can install it by following the instructions on the Docker website: https://docs.docker.com/get-docker/

1. Get nvidia image for Cuda 11.6
    ```console
    docker pull nvidia/cuda:11.6.2-base-ubuntu20.04
    ```
2. Build docker Image 
    ```console
    docker build -t translation-service .
    ```
3. Run Image
    ```console 
    docker run --gpus all -p 8080:8080 translation-service
    ```
4. Optional: Upload your Image to the Docker Hub
