FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:0 \
    VGL_REFRESHRATE=60 \
    VGL_ISACTIVE=1 \
    VGL_DISPLAY=egl \
    VGL_WM=1

# 3. Updated System Dependencies (Added zlib and SDL2 for retro)
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales wget bzip2 ca-certificates build-essential byobu curl git-core \
    htop pkg-config python3-pip python3-dev python3-gi python3-setuptools \
    python3-wheel unzip freeglut3-dev xvfb \
    zlib1g-dev libsdl2-dev && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Install Miniforge (Native ARM64)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    /opt/conda/bin/conda init bash

# 5. Fixed Environment Build
# We force a specific CMake version and inject the Policy flag
RUN mamba create -n dev python=3.10 -y && \
    mamba install -n dev -y numpy pytorch tianshou "cmake<3.25" ninja \
        opencv tensorboard -c conda-forge && \
    /opt/conda/bin/conda run -n dev /bin/bash -c \
    "export CMAKE_ARGS='-DCMAKE_POLICY_VERSION_MINIMUM=3.5' && \
     pip install --no-cache-dir git+https://github.com/MatPoliquin/stable-retro.git" && \
    conda clean -afy

WORKDIR /code
ENTRYPOINT ["/bin/bash"]