FROM nvcr.io/nvidia/caffe:19.08-py2

RUN bash -c "pip install easydict"
RUN bash -c "mkdir -p ./data/faster_rcnn_models/ && cd ./data/faster_rcnn_models && wget https://www.dropbox.com/s/5xethd2nxa8qrnq/resnet101_faster_rcnn_final.caffemodel"

# build cython
#COPY ./lib /src/lib
#RUN bash -c "cd /src/lib && make"

# build caffe
#COPY ./caffe /src/caffe
#RUN bash -c "cd /src/caffe && make -j$(nproc) && make pycaffe"

# copy remaining source code
#COPY ./ /src/
RUN git clone https://github.com/ChenRocks/UNITER.git
RUN git clone https://github.com/ChenRocks/BUTD-UNITER-NLVR2
RUN bash -c "cd BUTD-UNITER-NLVR2/lib && make"
RUN bash -c "cd BUTD-UNITER-NLVR2/caffe && make -j$(nproc) && make pycaffe"

RUN bash -c "apt-get update && apt-get install -y python3-pip"
RUN bash -c "pip3 install  numpy"
RUN bash -c "pip3 install  torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"

RUN bash -c "pip3 install pytorch-pretrained-bert==0.6.2 msgpack-numpy  cytoolz==0.11"
RUN bash -c "git clone https://github.com/NVIDIA/apex"
RUN bash -c "cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./"

RUN bash -c "pip3 install flask Flask-WTF"
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

