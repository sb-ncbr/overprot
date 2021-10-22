FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install -y sudo

# Unnecessary (installed within install.sh)
# RUN apt-get install -y nginx
# RUN apt-get install -y redis-server
# RUN apt-get install -y python-is-python3
# RUN apt-get install -y python3-venv
# RUN apt-get install -y pymol

RUN mkdir -p /server/overprot
WORKDIR /server/overprot

RUN mkdir software
RUN mkdir data
RUN mkdir data/data
RUN mkdir data/jobs
RUN mkdir data/logs

# COPY . software/overprot
# RUN bash software/overprot/OverProt/install.sh --clear
# RUN bash software/overprot/OverProtServer/install.sh --clear
COPY OverProt software/overprot/OverProt
RUN bash software/overprot/OverProt/install.sh --clear
COPY OverProtServer/install.sh software/overprot/OverProtServer/install.sh
COPY OverProtServer/requirements.txt software/overprot/OverProtServer/requirements.txt
RUN bash software/overprot/OverProtServer/install.sh --clear
COPY OverProtServer software/overprot/OverProtServer

# RUN mkdir nginx
# COPY trying-docker/nginx.conf nginx/nginx.conf
# COPY trying-docker/message.txt data/message.txt
# ENTRYPOINT ["nginx", "-c", "/server_data/nginx/nginx.conf", "-g", "daemon off;"]

ENV N_RQ_WORKERS=8
ENV N_GUNICORN_WORKERS=4

ENTRYPOINT ["bash", "/server/overprot/software/overprot/OverProtServer/startup-docker.sh"]

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