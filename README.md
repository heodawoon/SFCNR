# Deep Learning-based Brain Age Prediction in Patients with Schizophrenia Spectrum Disorders
This repository provides the PyTorch implementation of the utilized SFCNR framework for brain age prediction.


## Backbone Network
We utilized SFCN [1], the network proposed for brain age prediction, for a backbone network.
- Peng, H. et al., (2021). UKBiobank_deep_pretrain [Source code]. GitHub. [https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain)


## Overall Framework
![image](https://github.com/heodawoon/SFCNR/blob/main/image/OverallFramework_SFCNR.jpg)


## Datasets
1. Pre-training
The dataset utilized for the model's pre-training was sourced from the UK Biobank database [2], which is available at [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/). From a cumulative total of 49,123 T1 structural brain scans, the imaging datasets from the first visits of healthy participants were selected. Specifically, 230 subjects were randomly from each year in the age range of 48 and 80, to ensure a uniform age distribution within the dataset. Consequently, a total of 7,590 datasets were gathered, and these data were subsequently divided into training, validation, and testing sets by 8:1:1 ratio.


2. Fine-tuning
The datasets utilized for the model's fine-tuning and validation processes were procured from multiple sources: Jeonbuk National University Hospital (JBUH), Korea University Anam Hospital (KUAH), and the Consortium for Reliability and Reproducibility (CoRR). The latter's dataset is accessible at [http://fcon_1000.projects.nitrc.org/indi/CoRR/html/index.html](http://fcon_1000.projects.nitrc.org/indi/CoRR/html/index.html).\
From the CoRR database, the participants' demographic for the study was specifically selected from the Asian population, with an age range between 18 and 75 years. The detailed dataset composition from the CoRR database is 56 subjects from Beijing Normal University, 100 subjects from the Institute of Psychology at the Chinese Academy of Sciences, 30 subjects from Jinling Hospital at Nanjing University, 70 subjects from Southwest University, and 22 subjects from Xuanwu Hospital, affiliated with Capital University of Medical Sciences.


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
The study was supported by Korean Mental Health Technology R&D Project, Ministry of Health and Welfare, Republic of Korea (HL19C0015), Korea Health Technology R&D Project through the Korea Health Industry Development Institute funded by the Ministry of Health and Welfare, Republic of Korea (HR18C0016), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-00079, Artificial Intelligence Graduate School Program (Korea University)).
