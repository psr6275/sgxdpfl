# sgxdpfl
Using SGX for Differentially Private Federated Learning. 

## Setting
1. Make virtual env
```bash
conda create -n sgxdpfl python=3.9
```

2. Install Libraries
```bash
conda activate sgxdpfl
pip install -r requirements.txt
```

3. make data directory
```bash
mkdir data
```
## Parameters
- refer to: https://github.com/wenzhu23333/Differential-Privacy-Based-Federated-Learning/blob/master/utils/options.py 
1. frac: the fraction of clients (in our case, we need to use frac =1.0 because we consider cross silo scenario)
2. num_users: the number of users

## Reference
1. DP-FL
- https://github.com/Yangfan-Jiang/Federated-Learning-with-Differential-Privacy 
- https://github.com/wenzhu23333/Differential-Privacy-Based-Federated-Learning/tree/master 
- https://github.com/maxencenoble/Differential-Privacy-for-Heterogeneous-Federated-Learning/tree/main/flearn 
