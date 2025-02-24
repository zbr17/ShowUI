cd /home/zbr/disk1/datasets/showui
mkdir ShowUI-desktop
cd ShowUI-desktop
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download showlab/ShowUI-desktop --repo-type dataset --local-dir .



cd /home/zbr/disk1/datasets/showui
mkdir ShowUI-web
cd ShowUI-web
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download showlab/ShowUI-web --repo-type dataset --local-dir .



cd /home/zbr/disk1/datasets/showui
mkdir AMEX
cd AMEX
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Yuxiang007/AMEX --repo-type dataset --local-dir .



cd /home/zbr/disk1/datasets/showui
mkdir ScreenSpot
cd ScreenSpot
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download KevinQHLin/ScreenSpot --repo-type dataset --local-dir .



cd /home/zbr/disk1/datasets/showui
mkdir GUIAct
cd GUIAct
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download yiye2023/GUIAct --repo-type dataset --local-dir .

