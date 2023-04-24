FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN conda create -n proj_docker python=3.7

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "proj_docker", "/bin/bash", "-c"]

RUN conda install opencv=4.6.0 matplotlib=3.5.2 scipy=1.7.3 pandas=1.3.5 pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cpuonly=2.0 -c pytorch -c conda-forge
RUN conda install scikit-image=0.19.3
RUN conda install tqdm=4.64.1
# The code to run when container is started:
COPY app .
COPY cache ../root
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "proj_docker", "python", "check.py"]