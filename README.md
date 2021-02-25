### Installation
First clone repository, open a terminal and cd to the repository
    
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    mkdir -p data/raw/  # creates directories for data
    mkdir -p data/transformed
    mkdir -p data/models
    

For downstream finetuning, you also need to install torch, pytorch-transformers package and APEX (here for CUDA 10.0, which
is compatible with torch 1.1.0 ). You can also perform downstream finetuning without APEX, but it has been used for the paper.

    pip install scipy sckit-learn  # pip install --default-timeout=100 scipy; if you get a timeout
    pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
    pip install pytorch-transformers tensorboardX

    cd ..
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    
## Down-Stream Classification

Check the README in the "finetuning_and_classification" folder for how to train the BERT-ADA models
on the downstream task.
