# LPPN: a fast phase picking network

## Introduction

### Summary
We here present one lightweight phase picking network (LPPN) to pick P/S phases from continuous seismic recordings. It first classifies the phase type for consecutive data points with stride S, and then performs regression for accurate phase arrival time. The classification-regression approach and model optimization using deep separable convolution reduce the number of trainable parameters and ensure its computation efficiency. 

### Features
LPPN can be configured to have different model size and run on a wide range of devices. Here are:
- S can be configure as 8, 16, 32, 64, 128 
- n can be any size.

## Requirements
Python>=3.6.0 is required including PyTorch>=1.7

## Usage
### Data 
The STEAD can be obtained from https://github.com/smousavi05/STEAD. After download the data, then copy the .csv and .h5 file to the data folder. 

### Training
Running lppntrain.py will train the model:
```bash 
python lppntrain.py -f num_of_feature -s num_of_stride -l learning_rate -i data_folder -d device
```
where 
- num_of_feature is the nubmer of features. 
- num_of_stride is the total stride
- learning_rate is the learning rate during training the model
- data_folder is the STEAD data folder 
- device is the running device. 

### Infering
Running lppnvalid.py will valid the model:
```bash 
python lppnvalid.py -f num_of_feature -s num_of_stride -l learning_rate -o output -i data_folder -d device
```
the output file is num_of_feature-num_of_stride.stat.txt in output folder. 


## Contact 
cangye@hotmail.com

## Authors
Ziye Yu, cangye@hotmail.com

Weitao Wang, ...

## License
MIT 
