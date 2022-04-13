# Build: docker build .. -f Dockerfile -t overprot-core
# Run:   docker run -it -v /host/data/dir:/data overprot-core

FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install -y sudo
RUN apt-get install -y python-is-python3

COPY OverProtCore/install.sh /OverProtCore/install.sh
COPY OverProtCore/dotnet-install.sh /OverProtCore/dotnet-install.sh
COPY OverProtCore/requirements.txt /OverProtCore/requirements.txt
RUN bash /OverProtCore/install.sh
ENV PATH="/OverProtCore/venv/bin:$PATH"

COPY OverProtCore/ /OverProtCore/
COPY LICENSE /OverProtCore/LICENSE
WORKDIR /OverProtCore/

ENTRYPOINT ["bash"]
