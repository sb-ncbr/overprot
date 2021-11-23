# OverProt Server

**OverProt Server** allows using **OverProt** as a web application without need to install locally. It can also serve precomputed OverProt results.

## Running OverProt Server

1. Install Docker on your server machine:

    ```bash
    sudo apt-get update -y
    sudo apt-get install docker docker.io docker-compose
    sudo usermod -aG docker $USER
    ```

    Log out and log in again

2. Pull the Docker image:

    ```bash
    docker pull registry.gitlab.com/midlik/overprot
    ```

3. Copy `docker-compose.yaml` to your server machine and change the settings appropriately (set ports, volume mount-points...).

4. Start the container:

    ```bash
    docker-compose -f PATH_TO_DOCKER_COMPOSE_YAML up -d
    docker logs overprot_overprot_server_1 
    # docker-compose -f PATH_TO_DOCKER_COMPOSE_YAML down  # stop the container
    ```

## Running in development mode

Running in development mode (without Docker, Nginx, Gunicorn):

```bash
bash ../OverProtCore/install.sh
bash install.sh
bash startup-dev.sh  # Set path in this script properly
```
