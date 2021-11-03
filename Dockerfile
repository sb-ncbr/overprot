FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install -y sudo

RUN mkdir -p /server
WORKDIR /server

RUN mkdir bin
RUN mkdir var
RUN mkdir var/jobs
RUN mkdir var/logs
RUN mkdir data

# COPY . bin/overprot
# RUN bash bin/overprot/OverProt/install.sh --clear
# RUN bash bin/overprot/OverProtServer/install.sh --clear
COPY OverProt bin/overprot/OverProt
RUN bash bin/overprot/OverProt/install.sh --clear
COPY OverProtServer/install.sh bin/overprot/OverProtServer/install.sh
COPY OverProtServer/requirements.txt bin/overprot/OverProtServer/requirements.txt
RUN bash bin/overprot/OverProtServer/install.sh --clear
COPY OverProtServer bin/overprot/OverProtServer

# RUN mkdir nginx
# COPY trying-docker/nginx.conf nginx/nginx.conf
# COPY trying-docker/message.txt var/message.txt
# ENTRYPOINT ["nginx", "-c", "/server_data/nginx/nginx.conf", "-g", "daemon off;"]

ENV N_RQ_WORKERS=8
ENV N_GUNICORN_WORKERS=4
ENV HTTP_PORT=80

ENTRYPOINT ["bash", "/server/bin/overprot/OverProtServer/startup-docker.sh"]

# TODO fix redis-queue not starting (Error 99 connecting to localhost:6379. Cannot assign requested address.)






# DIR=$(dirname $0)
# cd $DIR

# python3 -m venv venv
# . venv/bin/activate
# python3 -m pip install -r requirements.txt
# mv venv/pyvenv.cfg venv/pyvenv-orig.cfg
# sed 's/include-system-site-packages *= *false/include-system-site-packages = true/' venv/pyvenv-orig.cfg > venv/pyvenv.cfg  # because pymol cannot be installed via pip


# RUN mkdir -p /data/PDBe_clone_binary /data/PDBe_clone

# RUN touch /data/pivots

# RUN mkdir -p /var/local/ProteinSearch/computations

# RUN chown -R apache:apache /var/local/ProteinSearch

# RUN mkdir -p /usr/local/www/ProteinSearch

# COPY docker/ProteinSearch.conf /etc/httpd/conf.d/

# WORKDIR /usr/src

# # set --build-arg REBUILD=$(date +%s) to rebuild image from here
# ARG REBUILD=unknown

# RUN git clone https://github.com/krab1k/gesamt_distance

# WORKDIR /usr/src/gesamt_distance

# RUN mkdir build

# WORKDIR /usr/src/gesamt_distance/build

# RUN cmake ..

# RUN make -j7

# RUN make install

# COPY app/ /usr/local/www/ProteinSearch/

# COPY docker/config.py /usr/local/www/ProteinSearch/

# ENTRYPOINT ["/usr/sbin/httpd", "-D", "FOREGROUND"]

# EXPOSE 8888