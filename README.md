# SSTBAN model
We give three examples (PEMSD8_36_36, PEMSD4_36_36, Seattle_1hour_36_36), and hyperparameters in the configuration documents are the best.


- ## PEMSD8 experiments:
  - ### data prepare:
  - 1. Enter the "SSTBAN/prepare/" 
  - 2. Run "python prepareData_SSTBAN.py --config  configurations/PEMSD8_1dim_construct_samples.conf"
  -    3.    We can get different training data by adjusting the PEMSD8_1dim_construct_samples.conf
  - ### model training:
  - 1. Enter the "SSTBAN/"
  - 2. Run "python train_SSTBAN.py --config configurations/PEMSD8_1dim_24.conf"

- ## PEMSD4 experiments:
  - ### data prepare:
  - 1. Enter the "SSTBAN/prepare/" 
  - 2. Run "python prepareData_SSTBAN.py --config  configurations/PEMSD4_1dim_construct_samples.conf"
  -    3.    We can get different training data by adjusting the PEMSD4_1dim_construct_samples.conf
  - ### model training:
  - 1. Enter the "SSTBAN/"
  - 2. Run "python train_SSTBAN.py --config configurations/PEMSD4_1dim_24.conf"


- ## Seattle experiments:
  - ### data prepare:
  - 1. Enter the "SSTBAN/prepare/" 
  - 2. Run "python prepareData_SSTBAN.py --config  configurations/Seattle_3dim_construct_samples.conf"
  -    3.    We can get different training data by adjusting the Seattle_3dim_construct_samples.conf
  - ### model training:
  - 1. Enter the "SSTBAN/"
  - 2. Run "python train_SSTBAN.py --config configurations/Seattle_3dim_24.conf"
  
You can download the data from:

Google Drive:
https://drive.google.com/file/d/1vBn8UDbIOz5T_MAahVNoYccR0jDjjFh-/view?usp=sharing

One Drive:
https://1drv.ms/u/s!AuEMUxDDxoXMxGdrVE7f9CYy330R?e=l6Jy01

Baidu Cloud:
https://pan.baidu.com/s/1MwWspHYSsg5fJsiYMiqn2g     password：g4hw

Aliyun Cloud:
https://www.aliyundrive.com/s/1bkb3bUFE53 



