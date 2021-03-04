FROM nvcr.io/nvidia/cuda:11.2.1-devel-ubuntu20.04 as ffmpeg-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV NVIDIA_DRIVER_CAPABILITIES=all


# https://github.com/NVIDIA/nvidia-docker/issues/969
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y curl git pkg-config yasm libx264-dev checkinstall \
 && rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers for FFmpeg NVIDIA hardware acceleration
RUN cd /tmp \
 && git clone --branch n9.0.18.1 --single-branch --depth 1 \
    https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
 && cd nv-codec-headers \
 && make -j$(nproc) && make install \
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
 && make -j$(nproc) \
 && checkinstall -y --nodoc --install=no \
 && mv ffmpeg_$FFMPEG_VERSION-1_amd64.deb /ffmpeg.deb \
 && cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION


################################################################################


FROM nvcr.io/nvidia/cuda:11.2.1-devel-ubuntu20.04 as tvl-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV NVIDIA_DRIVER_CAPABILITIES=all


# https://github.com/NVIDIA/nvidia-docker/issues/969
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y curl git swig python3.8-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
 && apt-get install -y pkg-config \
 && rm -rf /var/lib/apt/lists/*

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

# Install CMake
RUN pip3 install cmake==3.18.4

# Install scikit-build
RUN pip3 install scikit-build==0.10.0

RUN mkdir /app
WORKDIR /app

COPY . /app
RUN make dist -j$(nproc)


################################################################################

FROM nvcr.io/nvidia/cuda:11.2.1-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV NVIDIA_DRIVER_CAPABILITIES=all

# https://github.com/NVIDIA/nvidia-docker/issues/969
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
 && apt-get install -y curl git \
 && rm -rf /var/lib/apt/lists/*



RUN apt-get update \
 && apt-get install -y pkg-config \
 && rm -rf /var/lib/apt/lists/*

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
 && apt-get install -y libsm6 libxext6 libxrender1 python3.8-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

RUN pip3 install --upgrade jax jaxlib==0.1.61+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install tvl
COPY --from=tvl-builder /app/dist/tvl*.whl /tmp/
RUN pip3 install -f /tmp \
    tvl \
    tvl-backends-fffr \
  && rm /tmp/tvl*.whl

COPY . /app

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

CMD ["make", "test"]
