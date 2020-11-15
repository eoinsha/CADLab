FROM python:3.8-buster

RUN mkdir /src
WORKDIR /src

COPY requirements.txt /src
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY trained_models_nih/ /src/trained_models_nih/
COPY *.py /src/

ENTRYPOINT ["python3", "-m", "run"]
