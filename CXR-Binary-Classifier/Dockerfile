FROM continuumio/miniconda

RUN mkdir /src
WORKDIR /src

COPY environment.yml /src
RUN conda env create -f environment.yml
COPY trained_models_nih/ /src/trained_models_nih/
COPY *.py /src/

ENTRYPOINT ["conda", "run", "-n", "cxr-bin-cls", "python3", "-m", "run"]
