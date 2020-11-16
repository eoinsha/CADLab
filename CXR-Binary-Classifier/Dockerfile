FROM python:3.8-buster

RUN mkdir /src
WORKDIR /src

COPY requirements.txt /src
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY trained_models_nih/ /src/trained_models_nih/
ENV TORCH_HOME=/var/torch
RUN mkdir -p /var/torch/checkpoints
RUN curl https://download.pytorch.org/models/densenet121-a639ec97.pth > /var/torch/checkpoints/densenet121-a639ec97.pth
COPY *.py /src/

ENTRYPOINT ["python3", "-m", "run"]
