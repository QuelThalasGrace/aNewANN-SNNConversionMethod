# aNewANN-SNNConversionMethod
This is a project for our paper A New ANN-SNN Conversion Method with High Accuracy, Low Latency and Good Robustness in IJCAI2023

### Tips:
1. Since it involves another article being submitted, we only provide the code of the ANN part for the time being, and we will provide the code for the conversion part later.
2. The trained parameter file is too large to be uploaded. If any readers need to reproduce our work, please contact us through the email in the paper.

### About the training process
Our idea is mainly divided into two steps, the first step is to train the corresponding ANNs based on the StepReLU activation function, and the second step is to load the trained parameter files to the SNNs. For details of the StepReLU activation function, you can refer to the relevant code in the ann_StepReLU.py file.

You can also experiment on other network structures, but we only provide the training process of the resnet18 network in the train.py file. When you do this, you just need to define the network structure in the usual way, then replace all ReLU functions in the network with StepReLU functions, and specify the corresponding parameters. The util file mainly contains data enhancement methods for different data sets and methods for adding noise. When you use it, you can call it directly in the transform method.
~~~python
import numpy
