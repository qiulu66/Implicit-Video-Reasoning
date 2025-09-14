pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

conda install -y gcc_linux-64 gxx_linux-64
ln -s $CONDA_PREFIX/bin/x86_64-conda_cos7-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
ln -s $CONDA_PREFIX/bin/x86_64-conda_cos7-linux-gnu-g++ $CONDA_PREFIX/bin/g++

# Install the packages in r1-v .
# cd src/r1-v 
cd /group/40101/luqiu/Implicit_CoT/Environment/Video-R1/src/r1-v
pip install -e ".[dev]"

# Addtional modulespip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install wandb==0.18.3
pip install tensorboardx
# pip install qwen_vl_utils torchvision
# pip install flash-attn --no-build-isolation
pip install /group/40101/luqiu/Implicit_CoT/Environment/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# vLLM support 
pip install vllm==0.7.3

pip install nltk
pip install rouge_score
pip install deepspeed

# fix transformers version
# pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
pip install transformers==4.52.4

cd /group/40101/luqiu/Implicit_CoT/Open-R1-Video/qwen-vl-utils
pip install -e .[decord]