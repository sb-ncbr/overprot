# Build: docker build .. -f Dockerfile -t overprot-server
# Run:   docker-compose up [-d]

FROM ubuntu:22.04

RUN apt-get update -y

# # Debug:
# RUN apt-get install -y wget iputils-ping iputils-tracepath traceroute less

RUN mkdir -p /srv
WORKDIR /srv

RUN mkdir -p bin
RUN mkdir -p var
RUN mkdir -p var/jobs
RUN mkdir -p var/logs
RUN mkdir -p data

COPY OverProtCore/ bin/overprot/OverProtCore/
RUN bash bin/overprot/OverProtCore/install.sh --clean --no-sudo

COPY OverProtServer/install.sh bin/overprot/OverProtServer/install.sh
COPY OverProtServer/requirements.txt bin/overprot/OverProtServer/requirements.txt
RUN bash bin/overprot/OverProtServer/install.sh --clean --no-sudo
COPY OverProtServer/ bin/overprot/OverProtServer/
COPY LICENSE bin/overprot/LICENSE

COPY OverProtServer/init_var/ init_var/

ENV N_RQ_WORKERS="8"
ENV N_GUNICORN_WORKERS="4"
ENV HTTP_PORT="80"
ENV HTTPS_PORT=""
ENV OVERPROT_STRUCTURE_SOURCE=""
ENV MAXIMUM_JOB_DOMAINS="500"
ENV JOB_TIMEOUT="86400"

ENTRYPOINT ["bash", "/srv/bin/overprot/OverProtServer/startup-docker.sh"]
