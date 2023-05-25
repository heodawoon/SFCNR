# Deep Learning-based Brain Age Prediction in Patients with Schizophrenia Spectrum Disorders
This repository provides the PyTorch implementation of the utilized SFCNR framework for brain age prediction.


## Backbone Network
We utilized SFCN [1], the network proposed for brain age prediction, for a backbone network.
- Peng, H. et al., (2021). UKBiobank_deep_pretrain [Source code]. GitHub. [https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain)


## Overall Framework
![image](https://github.com/heodawoon/SFCNR/blob/main/image/OverallFramework_SFCNR.jpg)


## Datasets
The data used for pre-training was sourced from the UK Biobank database [2], which is available at [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/). Out of a total database comprising 49,123 T1 structural brain scans, we used the first-visit imaging datasets of healthy subjects. In addition, we randomly picked 230 subjects from each year between 48 and 80 to maintain an even age distribution across the sample. This approach resulted in an overall selection of 7,590 data. These data were subsequently divided into training, validation, and testing sets by 8:1:1 ratio.


## Usage

- Pretraining
```
python ./SFCNR_pretrain.py --gpu_id 0
```

- Finetuning
```
python ./SFCNR_finetune.py --gpu_id 0
```


## References
[1] Peng et al., "Accurate brain age prediction with lightweight deep neural networks," Medical image analysis 68: 101871, 2021.\
[2] Sudlow et al., "UK biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age," PLoS medicine 12.3: e1001779, 2015.


## Acknowledgement
The corresponding author thanks all the study participants, as well as his father for guidance. The study was supported by Korean Mental Health Technology R&D Project, Ministry of Health and Welfare, Republic of Korea (HL19C0015) and Korea Health Technology R&D Project through the Korea Health Industry Development Institute funded by the Ministry of Health and Welfare, Republic of Korea (HR18C0016).
