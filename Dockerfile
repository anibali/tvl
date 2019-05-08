FROM nvidia/cuda:10.0-devel-ubuntu16.04 as ffmpeg-builder

RUN apt-get update \
 && apt-get install -y curl git pkg-config yasm libx264-dev checkinstall \
 && rm -rf /var/lib/apt/lists/*

# Install nv-codec-headers for FFmpeg NVIDIA hardware acceleration
RUN cd /tmp \
 && git clone --branch n9.0.18.1 --single-branch --depth 1 \
    https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
 && cd nv-codec-headers \
 && make && make install \
 && rm -rf /tmp/nv-codec-headers

# Build FFmpeg, enabling only selected features
ARG FFMPEG_VERSION=4.1.1
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
    --enable-libx264 \
    --enable-decoder=h264 \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,mxf \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --enable-filter=scale \
    --enable-ffnvcodec \
    --enable-nvenc \
 && make -j8 \
 && checkinstall -y --nodoc --install=no \
 && mv ffmpeg_$FFMPEG_VERSION-1_amd64.deb /ffmpeg.deb \
 && cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION


################################################################################


FROM nvidia/cuda:10.0-devel-ubuntu16.04 as tvl-builder

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
RUN conda install -y swig=3.0.12 && conda clean -ya

# Add a stub version of libnvcuvid.so for building (required for CUDA backends).
# This library is provided by nvidia-docker at runtime when the environment variable
# NVIDIA_DRIVER_CAPABILITIES includes "video".
RUN curl -so /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 \
    https://raw.githubusercontent.com/NVIDIA/nvvl/bde20830cf171af8d10ef8222449237382b178ef/pytorch/test/docker/libnvcuvid.so \
 && ln -s /usr/local/nvidia/lib64/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so \
 && ln -s libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

# Install nvvl
RUN pip install cmake==3.13.3
RUN cd /tmp \
 && git clone https://github.com/NVIDIA/nvvl.git \
 && cd nvvl \
 && git checkout 14b660c87b4c5a86d95da04a50be70674e68e625 \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make -j8 \
 && make install

RUN mkdir /app
WORKDIR /app

COPY . /app
RUN make dist


################################################################################


FROM nvidia/cuda:10.0-devel-ubuntu16.04

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
COPY --from=tvl-builder /usr/local/lib/libnvvl.so /usr/local/lib/
COPY --from=tvl-builder /app/dist/tvl*.whl /tmp/
RUN pip install -f /tmp \
    tvl \
    tvl-backends-nvdec \
    tvl-backends-nvvl \
    tvl-backends-opencv \
    tvl-backends-pyav \
  && rm /tmp/tvl*.whl

COPY . /app

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

CMD ["make", "test"]
