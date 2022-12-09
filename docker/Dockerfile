# Ubuntu 18.04, Python 3.7, PyTorch 1.12.1, CUDA 11.3
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

### nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

### change the system source for installing libs
RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
    echo "Use aliyun source for installing libs"
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN rm /etc/apt/sources.list.d/cuda.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

### timezone
RUN apt-get install tzdata && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Install MMEngine , MMCV and MMDet
RUN pip install --no-cache-dir openmim && \
    mim install --no-cache-dir "mmengine>=0.3.1" "mmcv>=2.0.0rc1,<2.1.0" "mmdet>=3.0.0rc3,<3.1.0"

# Use MMYOLO as a dependency or third-party package, install it with MIM
RUN mim install --no-cache-dir "mmyolo"

# Albumentation
RUN pip install albumentations --no-binary qudida,albumentations

### clean all cache
RUN pip cache purge && rm -rf /var/lib/apt/lists/* && apt-get clean
COPY ./Dockerfile /home/${USER}
