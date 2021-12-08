FROM nvidia/cuda:10.0-devel-ubuntu18.04 as ffmpeg-builder

# https://github.com/NVIDIA/nvidia-docker/issues/969
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y curl git pkg-config yasm libx264-dev checkinstall \
 && rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers for FFmpeg NVIDIA hardware acceleration
RUN cd /tmp \
 && git clone --branch n9.0.18.1 --single-branch --depth 1 \
    https://github.com/FFmpeg/nv-codec-headers.git \
 && cd nv-codec-headers \
 && make && make install \
 && rm -rf /tmp/nv-codec-headers

# Build FFmpeg, enabling only selected features
ARG FFMPEG_VERSION=4.2.2
RUN cd /tmp && curl -sLO http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 \
 && tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 \
 && rm ffmpeg-$FFMPEG_VERSION.tar.bz2 \
 && cd ffmpeg-$FFMPEG_VERSION \
 && ./configure --prefix=/usr/local \
    --disable-static \
    --enable-shared \
    --enable-gpl \
    --disable-iconv \
    --disable-doc \
    --disable-ffplay \
 && make -j8 \
 && checkinstall -y --nodoc --install=no \
 && mv ffmpeg_$FFMPEG_VERSION-1_amd64.deb /ffmpeg.deb \
 && cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION


################################################################################


FROM nvidia/cuda:10.0-devel-ubuntu18.04 as tvl-builder

# https://github.com/NVIDIA/nvidia-docker/issues/969
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y curl git \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.6.13
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.13 \
 && conda clean -ya

RUN apt-get update \
 && apt-get install -y pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Install project requirements
COPY environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml \
 && conda clean -ya \
 && rm /tmp/environment.yml

# Install FFmpeg
COPY --from=ffmpeg-builder /ffmpeg.deb /tmp/ffmpeg.deb
RUN dpkg -i /tmp/ffmpeg.deb && rm /tmp/ffmpeg.deb

# Add a stub version of libnvcuvid.so for building (required for CUDA backends).
# This library is provided by nvidia-docker at runtime when the environment variable
# NVIDIA_DRIVER_CAPABILITIES includes "video".
RUN curl -sLo /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 \
    https://raw.githubusercontent.com/NVIDIA/nvvl/bde20830cf171af8d10ef8222449237382b178ef/pytorch/test/docker/libnvcuvid.so \
 && ln -s /usr/local/nvidia/lib64/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so \
 && ln -s libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

RUN mkdir /app
WORKDIR /app

COPY . /app
RUN make dist


################################################################################


FROM nvidia/cuda:10.0-devel-ubuntu18.04

# https://github.com/NVIDIA/nvidia-docker/issues/969
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y curl git \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.6.13
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.6.13 \
 && conda clean -ya

RUN apt-get update \
 && apt-get install -y pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Install project requirements
COPY environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml \
 && conda clean -ya \
 && rm /tmp/environment.yml

# Install X display server client
RUN apt-get update \
 && apt-get install -y libx11-6 \
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

# Install tvl
COPY --from=tvl-builder /app/dist/tvl*.whl /tmp/
RUN pip install -f /tmp \
    tvl \
    tvl-backends-nvdec \
    tvl-backends-opencv \
    tvl-backends-pyav \
    tvl-backends-fffr \
  && rm /tmp/tvl*.whl

COPY . /app

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

CMD ["make", "test"]
