FROM python:3.7-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_XLA_FLAGS "--tf_xla_cpu_global_jit"

WORKDIR pydl

COPY requirements.txt .

RUN pip install pip -U && \
    pip3 install --no-cache-dir -r requirements.txt -U && \
    pip3 install --no-cache-dir tensorflow

ADD . .

RUN python3 setup.py install -O2

WORKDIR /
RUN rm -rf pydl