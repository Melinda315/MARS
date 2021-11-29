# MARS
This repository contains a implementation of our "Multi-Facet Recommender Networks with Spherical Optimization" accepted by ICDE 2021.

## Contact

Please contact us if you have problems with the code, and also if you think your work is relevant but missing from the survey.

Yanchao Tan (yctan@zju.edu.cn), Xiangyu Wei (weixy@zju.edu.cn)

## Environment Setup
1. Pytorch 1.2+
2. Python 3.6+

## Guideline

### data

We provide a dataset ciao , which contains:

- All interactions of the dataset(```ratings.dat```);
- Train set, validation set, and test set devided by drop the last two items of each user from ratings.dat(```LOOTrain.dat```, ```LOOTest.dat```, ```LOOVal.dat```);
- 100 unordered items for each user for testing(```LOONegatives.dat```);
- 200 pre-sampled data to accelerate the speed of training in 200 epochs(```samples/sampling_*.dat```);

### model

The implementation of MARS(```model.py```); 

### utils

Data input and model evaluation

- Dataset.py For Data preprocessing;

- Recommender.py The base class of models, including functions of getting training and testing instances and evaluating performances;

- evaluation.py Functions of calculating NDCG and HR;

## Example to run the codes
```
python main.py --dataset ciao --numEpoch 100 --lRate 0.01
```

### Citation
If you find the software useful, please consider citing the following paper:
```
@inproceedings{tan2021multi,
  title={Multi-Facet Recommender Networks with Spherical Optimization},
  author={Tan, Yanchao and Yang, Carl and Wei, Xiangyu and Ma, Yun and Zheng, Xiaolin},
  booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)},
  pages={1524--1535},
  year={2021},
  organization={IEEE}
}
```
