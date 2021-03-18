FROM ubuntu:20.04

RUN apt install -y python3-venv nginx redis-server pymol

RUN mkdir /server_data

WORKDIR /server_data

RUN mkdir data
RUN mkdir jobs
RUN mkdir logs

# TODO 






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