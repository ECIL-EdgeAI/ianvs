[Links of scenarios]: ../proposals/scenarios/
[the PCB-AoI public dataset]: https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi
[details of PCB-AoI dataset]: ../proposals/scenarios/industrial-defect-detection/pcb-aoi.md
[XFTP]: https://www.xshell.com/en/xftp/
[FPN-model]: https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/model.zip

# Quick Start

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, 
in order to facilitate more efficient and effective development. Quick start helps you to test your algorithm on Ianvs 
with a simple example on industrial defect detection. You can reduce manual procedures to just a few steps so that you can 
building and start your distributed synergy AI solution development within minutes. 

Before using Ianvs, you might want to have the device ready: 
- one machine is all you need, i.e., a laptop or a virtual machine is sufficient and cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- internet connection for github and pip, etc
- Python 3.6+ installed
  
In this example, we are using Linux platform with Python 3.6.9. If you are using Windows, most steps should still apply but a few like commands and package requirements might be different. 

## Step 1. Ianvs Preparation

First, we download the code of Ianvs. Assuming that we are using `/home/ianvs-qs` as workspace, Ianvs can be cloned with `Git` as:        
``` shell
/home$ cd /home/ianvs-qs #One might use other path preferred

/home/ianvs-qs$ mkdir -p ./project/
/home/ianvs-qs$ cd ./project/
/home/ianvs-qs/project$ git clone https://github.com/kubeedge/ianvs.git   
```
<!-- https://github.com/JimmyYang20/ianvs.git -->


Then, we install third-party dependencies for ianvs. 
``` shell
/home/ianvs-qs/project$ cd ./ianvs 

/home/ianvs-qs/project/ianvs$ apt update
/home/ianvs-qs/project/ianvs$ apt install libgl1-mesa-glx -y
/home/ianvs-qs/project/ianvs$ python -m pip install --upgrade pip

/home/ianvs-qs/project/ianvs$ python -m pip install third_party/*
/home/ianvs-qs/project/ianvs$ python -m pip install -r requirements.txt
```

We are now ready to install Ianvs. 
``` shell
/home/ianvs-qs/project/ianvs$ python setup.py install  
```

## Step 2. Dataset and Model Preparation 
  
Datasets and models can be large. To avoid over-size projects in the Github repository of Ianvs, the Ianvs code base do not include origin datasets and models. Then developers do not need to download non-necessary datasets and models for a quick start.

First, the user need to prepare the dataset according to the targeted scenario, from source links (e.g., from Kaggle) provided by Ianvs. Scenarios with dataset are  available [Links of scenarios]. As an example in this document, we are using [the PCB-AoI Public Dataset] released by KubeEdge SIG AI members on Kaggle. See [details of PCB-AoI dataset] for more information of this dataset. 



``` shell
/home/ianvs-qs/project/ianvs$ cd /home/ianvs-qs #One might use other path preferred
/home/ianvs-qs$ mkdir -p ./dataset/   
```

Please put the downloaded dataset on the above datset path, e.g., `/home/ianvs-qs/dataset`. One can transfer the dataset to the path, e.g., on a remote Linux system using [XFTP]. 



``` shell
/home/ianvs-qs$ cd ./dataset/  
/home/ianvs-qs/dataset$ tar -zxvf pcb_imgs.tar.gz
```

Then we may Develop the targeted algorithm as usual. In this quick start, Ianvs has prepared an initial model for benchmarking. One can find the model at [FPN-model].



``` shell
/home/ianvs-qs/dataset$ cd /home/ianvs-qs #One might use other path preferred
/home/ianvs-qs$ mkdir -p ./initial_model  
```

Please put the downloaded model on the above model path, e.g., `/home/ianvs-qs/initial_model`. One can transfer the model to the path, e.g., on remote a Linux system using [XFTP]. 

Related algorithm is also ready as a wheel in this quick start. 
``` shell
/home/ianvs-qs/initial_model$ cd /home/ianvs-qs #One might use other path preferred
/home/ianvs-qs$ cd ./project/ianvs/
/home/ianvs-qs/project/ianvs$ python -m pip install examples/resources/algorithms/FPN_TensorFlow-0.1-py3-none-any.whl
```


## Step 3. Ianvs Execution and Presentation

We are now ready to run the ianvs for benchmarking on PCB-AoI dataset. 

``` shell
/home/ianvs-qs/project/ianvs$ ianvs -f examples/pcb-aoi/singletask_learning_bench/benchmarkingjob.yaml
```

Finally, the user can check the result of benchmarking on the console and also in the output path(e.g. `/ianvs/singletask_learning_bench/workspace`) defined in the
benchmarking config file (e.g. `benchmarkingjob.yaml`), which might look like:   

|rank|algorithm              |f1_score          |paradigm          |basemodel|learning_rate|momentum|time               |url                                                                                                  |
|----|-----------------------|------------------|------------------|---------|-------------|--------|-------------------|-----------------------------------------------------------------------------------------------------|
|1   |fpn_singletask_learning|0.9446112779446113|singletasklearning|estimator|0.1          |0.1     |2022-06-16 15:02:59|/ianvs/pcb-aoi/workspace/benchmarkingjob/fpn_singletask_learning/b3a84564-ed41-11ec-83c3-53ead20896e4|
