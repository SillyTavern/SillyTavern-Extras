FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

EXPOSE 5100

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv wget build-essential

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version

RUN conda init

RUN conda create -n extras

RUN /bin/bash -c "source activate extras"

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia -c conda-forge

WORKDIR /sillytavern-extras/
COPY . .

ARG REQUIREMENTS
RUN pip install -r $REQUIREMENTS

ARG MODULES
CMD ["python","server.py","--enable-modules=$MODULES"]
