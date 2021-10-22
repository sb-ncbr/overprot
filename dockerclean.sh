docker rm $(docker ps -aq)
yes | docker image prune