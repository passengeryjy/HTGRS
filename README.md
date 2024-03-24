# HTGRS
Code for "Document-level biomedical relation extraction via hierarchical tree graph and relation segmentation module"


## Requisites

- **Python >=3.8**
- **Pytorch1.13.0+cuda11.7**
- **Transformers = 4.37.2**

Example installation using conda:

```Python
#if your OS is Linux or Window, using the cuda command like this.
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Datasets

**The link addresses of the three Bio-DocRE data sets used in this article’s experiments are listed below.**

| Dataset | Link |
| ------- | ---- |
| CDR     |   https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip   |
| GDA     |   https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip   |
| BioRED  |   https://github.com/ncbi/BioRED   |


