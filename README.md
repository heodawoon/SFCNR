# Deep Learning-based Brain Age Prediction in Patients with Schizophrenia Spectrum Disorders

This repository provides the PyTorch implementation of the utilized SFCNR framework for brain age prediction.\
The code is based on the SFCN (Peng et al., 2021).


## Backbone Network
We utilized SFCN (Peng et al., 2021), the network proposed for brain age prediction, for a backbone network.
- Accurate brain age prediction with lightweight deep neural networks Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith Medical Image Analysis (2021); doi: https://doi.org/10.1016/j.media.2020.101871
- Peng, H. et al., (2021). UKBiobank_deep_pretrain [Source code]. GitHub. https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain


## Usage

- Pretraining
```
python ./SFCNR_pretrain.py --gpu_id 0
```

- Finetuning
```
python ./SFCNR_finetune.py --gpu_id 0
```
