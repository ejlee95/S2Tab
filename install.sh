# pytorch 2.0.1 cuda11.7 cudnn 8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

conda install cython scipy -y
pip install pycocotools

pip install tensorboardX tqdm jsonlines tensorboard imageio transformers==4.29.2 timm==0.9.2 opencv-python==4.9.0.80 distance apted lxml pytorch-metric-learning matplotlib
pip install omegaconf psutil PyMuPDF orjson

pip install numpy==1.26.4