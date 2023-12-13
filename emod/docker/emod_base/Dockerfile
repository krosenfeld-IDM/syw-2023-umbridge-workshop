# multi stage build...
# https://github.com/InstituteforDiseaseModeling/EMOD-Generic/blob/main/env_Ubuntu22/EMOD_EXE_Ubuntu22.def

FROM ubuntu:22.04 as deps_stage
RUN apt-get update \
    && apt-get install -y software-properties-common build-essential \
    libmpich-dev libsqlite3-dev \
    libboost-dev git scons python3-pip \
    python3.9 \
    && apt-get clean

# load EMOD files (accessed from Kurt's EMOD github)
# https://github.com/kfrey-idm/EMOD/tags
FROM ubuntu:22.04 as import_stage
WORKDIR /project
ADD v2.21.0.tar.gz .

FROM deps_stage as build_stage
WORKDIR /project/
# does this take just as long?
COPY --from=import_stage /project/EMOD-2.21.0/ .