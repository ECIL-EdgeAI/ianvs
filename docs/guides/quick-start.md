# Quick Start

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, 
in order to facilitate more efficient and effective development. Quick start helps you to test your algorithm on Ianvs 
with a simple example on industrial defect detection. You can reduce manual procedures to just a few steps so that you can 
building and start your distributed synergy AI solution development within minutes. 

The user flow for an algorithm developer is as follows. We will introduce these steps in the rest of this documents. 
![](images/user_flow.png)


## Step 0. Ianvs Preparation 
    
Before using Ianvs, you might want to have the device ready: 
- one machine is all you need, i.e., a laptop or a virtual machine is sufficient and cluster is not necessary
- 2 CPUs or more
- 2GB+ free memory, depends on algorithm and simulation setting
- 10GB+ freee disk space
- internet connection for github and pip, etc
- Python 3.x installed
  
BTW: In this quick start example, we adopt the ``Pycharm`` as Python development IDE on ``Windows``, which would be a common setting for algorithm developers helping to simplify package installation and usage which Ianvs required. If you are using other settings, most steps should still apply but a few might be different. 

Install required packages specified in requirement files. 
    
An algorithm developer can interpret/ compile the code of ianvs in his/her machine into executable files. 
        
## Step 1. Test Case Preparation 
  
Prepare the dataset according to the targeted scenario 
  
Justification: Datasets can be large. To avoid over-size projects in Github ianvs repository, the ianvs executable file and code base do not include origin datasets and developers can download datasets from source links (e.g., from Kaggle) given by ianvs. 

Leverage the ianvs algorithm interface for the targeted algorithm. 
Note that the tested algorithm should follow the ianvs interface to ensure functional benchmarking.
  
## Step 2. Algorithm Development

One may Develop the targeted algorithm as usual. 

## Step 3. ianvs Configuration

Fill configuration files for ianvs

## Step 4. ianvs Execution

Run the executable file of ianvs for benchmarking
   
## Step 5. ianvs Presentation

View the benchmarking result of the targeted algorithms
   
## Step 6. Repeat until Satisfactory

Step 3 - 6 until the targeted algorithm is satisfactory

