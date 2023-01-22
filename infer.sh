#!/bin/sh
source ${HOME}/workspace4/venv/activate_.sh

export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

nvcc -V

weights="./weights/yolact_plus_base_54_800000.pth"
inputs="./vinput/yolact_sample.mp4"
outputs="./voutput/sample_res.mp4"

python infer.py --trained_model=$weights \
                --config=yolact_plus_base_config \
                --score_threshold=0.3 \
                --top_k=10 \
                --video=$inputs:$outputs
