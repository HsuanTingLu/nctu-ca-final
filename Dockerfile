FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY strsort/ /strsort/
COPY fmindex/ /fmindex/
COPY scripts/ /scripts/
COPY CMakeLists.txt /

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]