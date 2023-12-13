# About

Dockerfile to create base Docker image for Generic EMOD (850Mb) for Ubuntu. This is used by the EMOD examples.

## Build on your machine

1. Install Docker

2. Download EMOD.tar.gz from https://github.com/kfrey-idm/EMOD/tags.  Note that for this build I've renamed the file and root folder to `EMOD/`.  

3. Build the image:

    docker build -t emod_base .

4. (optional) run:

    docker run -it emod_base