# Enviroments installation Instruciton
install the env in IsaacGymEnvs


install the dexgrasp_policy in UniDexGrasp
! Attention: 

In the setup.py, remove the following specific version 
``` python
    # "torch==1.13",
    # "torchvision==0.14",
    # "torchaudio==0.13",
    # "matplotlib==3.5.1",
    # "numpy==1.23.5",
```
pip install numpy==1.20

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

install pointnet2_pos_lib
```bash
git clone git@github.com:erikwijmans/Pointnet2_PyTorch.git
cd pointnet2_ops_lib/
python setup.py install
```
