FROM nvidia/cuda:10.0-devel-ubuntu14.04 as ffmpeg-builder

RUN apt-get update \
 && apt-get install -y curl git pkg-config yasm libx264-dev checkinstall \
 && rm -rf /var/lib/apt/lists/*

# Build FFmpeg, enabling only selected features
ARG FFMPEG_VERSION=3.4.5
RUN cd /tmp && curl -sO http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 \
 && tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 \
 && rm ffmpeg-$FFMPEG_VERSION.tar.bz2 \
 && cd ffmpeg-$FFMPEG_VERSION \
 && ./configure --prefix=/usr/local \
    --disable-static \
    --disable-everything \
    --disable-autodetect \
    --disable-iconv \
    --disable-doc \
    --disable-ffplay \
    --enable-shared \
    --enable-gpl \
    --enable-cuda \
    --enable-libx264 \
    --enable-decoder=h264 \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,mxf \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-filter=scale \
 && make -j8 \
 && checkinstall -y --nodoc --install=no \
 && mv ffmpeg_$FFMPEG_VERSION-1_amd64.deb /ffmpeg.deb \
 && cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION


################################################################################


FROM nvidia/cuda:10.0-devel-ubuntu14.04 as tvl-builder

RUN apt-get update \
 && apt-get install -y curl git \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.6.5
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.5 \
 && conda clean -ya

# Install PyTorch with CUDA support
RUN conda install -y -c pytorch \
    cuda100=1.0 \
    magma-cuda100=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1" \
 && conda clean -ya

RUN apt-get update \
 && apt-get install -y pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Install FFmpeg
COPY --from=ffmpeg-builder /ffmpeg.deb /tmp/ffmpeg.deb
RUN dpkg -i /tmp/ffmpeg.deb && rm /tmp/ffmpeg.deb

# Install Swig
RUN conda install -y swig=3.0.10 && conda clean -ya

# Add a stub version of libnvcuvid.so for building.
# This library is provided by nvidia-docker at runtime when the environment variable
# NVIDIA_DRIVER_CAPABILITIES includes "video".
RUN curl -so /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 \
    https://raw.githubusercontent.com/NVIDIA/nvvl/bde20830cf171af8d10ef8222449237382b178ef/pytorch/test/docker/libnvcuvid.so \
 && ln -s /usr/local/nvidia/lib64/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so \
 && ln -s libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
RUN python setup.py bdist_wheel \
 && mv dist/tvl-*.whl /


################################################################################


FROM nvidia/cuda:10.0-devel-ubuntu14.04

RUN apt-get update \
 && apt-get install -y curl git \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.6.5
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.5 \
 && conda clean -ya

# Install PyTorch with CUDA support
RUN conda install -y -c pytorch \
    cuda100=1.0 \
    magma-cuda100=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1" \
 && conda clean -ya

RUN apt-get update \
 && apt-get install -y pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Install FFmpeg
COPY --from=ffmpeg-builder /ffmpeg.deb /tmp/ffmpeg.deb
RUN dpkg -i /tmp/ffmpeg.deb && rm /tmp/ffmpeg.deb

RUN mkdir /app
WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update \
 && apt-get install -y libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN pip install -r requirements.txt

# Install tvl
COPY --from=tvl-builder /tvl-*.whl /tmp/
RUN pip install --no-index -f /tmp tvl && rm /tmp/tvl-*.whl

COPY . /app

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

CMD ["pytest", "-s"]
