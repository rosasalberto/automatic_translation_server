FROM nvidia/cuda:11.6.2-base-ubuntu20.04

RUN apt-get update && apt-get install -y curl python3-pip

COPY . /app
WORKDIR /app

RUN pip3 install pipenv
RUN pipenv install --system --deploy

EXPOSE 8080

CMD ["uvicorn", "server:app", "--reload","--port", "8080"]