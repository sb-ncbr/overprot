# OverProt Server

**OverProt Server** allows using **OverProt Core** as a web application without the need to install it locally. It can also serve precomputed OverProt results.
A running instance of OverProt Server is deployed at <https://overprot.ncbr.muni.cz>.

## Running OverProt Server

1. Install Docker on your server machine.

   E.g. on Ubuntu:

    ```bash
    sudo apt-get update -y
    sudo apt-get install docker docker.io docker-compose
    sudo usermod -aG docker $USER
    ```

    Log out and log in again.

2. Pull the Docker image:

    ```bash
    docker pull registry.gitlab.com/midlik/overprot/overprot-server
    docker images
    ```

3. Copy `docker-compose.yaml` to your server machine and change the settings appropriately (set ports, volume mount-points...).

4. Start the container:

    ```bash
    docker-compose -f PATH_TO_DOCKER_COMPOSE_YAML up -d
    docker ps -a  # Show the newly started container.
    docker logs CONTAINER_NAME  # Print logs to check if successful. 
    # docker-compose -f PATH_TO_DOCKER_COMPOSE_YAML down  # stop the container
    ```

## Running in development mode

Running in development mode (without Docker, Nginx, Gunicorn):

1. Set the paths in `startup-dev.sh` according to your system.

2. Run:

   ```bash
   bash ../OverProtCore/install.sh
   bash install.sh
   bash startup-dev.sh
   ```

## File organization within the Docker container

See `server_files.md`.

## Building the Docker image

Build:

```bash
docker build .. -f Dockerfile -t registry.gitlab.com/midlik/overprot/overprot-server
docker images
```

Test (will run on the host's `http://localhost:8080`):

```bash
docker-compose up
```

Push to the repository:

```bash
docker push registry.gitlab.com/midlik/overprot/overprot-server
```
