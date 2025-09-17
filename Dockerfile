# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS dev

# Install CUDA toolkit for development (optional - graceful fallback for non-GPU systems)
RUN --mount=type=cache,target=/var/cache/apt/ --mount=type=cache,target=/var/lib/apt/ \
    apt-get update && apt-get install --no-install-recommends --yes \
    wget gnupg2 software-properties-common && \
    (wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install --no-install-recommends --yes \
    cuda-toolkit-12-4 && \
    rm -f cuda-keyring_1.1-1_all.deb) || \
    echo "CUDA installation failed - continuing with CPU-only build"

# Create and activate a virtual environment [1].
# [1] https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Tell Git that the workspace is safe to avoid 'detected dubious ownership in repository' warnings.
RUN git config --system --add safe.directory '*'

# Create a non-root user and give it passwordless sudo access [1].
# [1] https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN --mount=type=cache,target=/var/cache/apt/ \
    --mount=type=cache,target=/var/lib/apt/ \
    groupadd --gid 1000 user && \
    useradd --create-home --no-log-init --gid 1000 --uid 1000 --shell /usr/bin/bash user && \
    chown user:user /opt/ && \
    apt-get update && apt-get install --no-install-recommends --yes \
    sudo clang libomp-dev build-essential cmake pkg-config && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user
USER user

# Configure the non-root user's shell.
RUN mkdir ~/.history/ && \
    echo 'HISTFILE=~/.history/.bash_history' >> ~/.bashrc && \
    echo 'bind "\"\e[A\": history-search-backward"' >> ~/.bashrc && \
    echo 'bind "\"\e[B\": history-search-forward"' >> ~/.bashrc && \
    echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Explicitly configure compilers for llama-cpp-python.
ENV CC=clang
ENV CXX=clang++

# Configure llama-cpp-python to use CUDA
ENV CMAKE_ARGS=-DLLAMA_CUDA=on
ENV FORCE_CMAKE=1
ENV CUDACXX=$CUDA_HOME/bin/nvcc
