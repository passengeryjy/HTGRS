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

## Project structure
If you want to reproduction this project, you should make the project sturcture like this.
```Python
HTGRS
 |-- dataset
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |    |--convert_CDR
 |    |    |    |-- convert_train.json
 |    |    |    |-- convert_dev.json 
 |    |    |    |-- convert_test.json
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |    |    |--convert_GDA
 |    |    |    |-- convert_train.json
 |    |    |    |-- convert_dev.json 
 |    |    |    |-- convert_test.json
 |    |-- biored
 |    |    |-- Dev.BioC_modified.JSON
 |    |    |-- Test.BioC_modified.JSON
 |    |    |-- Train.BioC_modified.JSON
 |-- saved_model
 |-- encoder
 |    |-- scibert_base
 |-- src
 |    |-- utils.py
 |    |-- adj_utils.py
 |    |-- convert_biored.py
 |    |-- convert_pro.py
 |    |-- long_seq.py
 |    |-- losses.py
 |    |-- run_cdr.py
 |    |-- run_gda.py
 |    |-- run_bio.py
 |    |-- rgcn.py
 |    |-- model.py
```
 #You can get the encoder you need from https://huggingface.co, such as https://huggingface.co/allenai/scibert_scivocab_cased

## Reproduction

Before you train the model, you should convert the corresponding dataset into the structure acceptable to our model.

- if you want to train the model after obtaining the features that model can accpet, you should set the arguments  "--save_path" to save your checkpoints. Then, you can use this command to train your model:

```Python
python run_*.py 
#like python run_cdr.py
```



- Evaluate the model on test set through setting the argument "--load_path" to your trained model:

```Python
python run_*.py
```

## Acknowledgement

We refer to the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) Thanks for their contributions.
