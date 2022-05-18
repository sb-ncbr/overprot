# Build: docker build .. -f Dockerfile -t overprot-core
# Run:   docker run -it -v /host/data/dir:/data overprot-core

FROM ubuntu:22.04

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3

COPY OverProtCore/install.sh /OverProtCore/install.sh
COPY OverProtCore/dotnet-install.sh /OverProtCore/dotnet-install.sh
COPY OverProtCore/requirements.txt /OverProtCore/requirements.txt  
RUN bash /OverProtCore/install.sh --no-sudo
ENV PATH="/OverProtCore/venv/bin:$PATH"

COPY OverProtCore/ /OverProtCore/
COPY LICENSE /OverProtCore/LICENSE
WORKDIR /OverProtCore/

ENTRYPOINT ["bash"]
