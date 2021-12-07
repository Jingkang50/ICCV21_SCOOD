# SCOOD-UDG (ICCV 2021)

[![paper](https://img.shields.io/badge/Paper-arxiv-b31b1b)](https://arxiv.org/abs/2108.11941)
&nbsp;
[![projectpage](https://img.shields.io/badge/Project%20Page-online-52b788)](https://jingkang50.github.io/projects/scood)
&nbsp;
[![gdrive](https://img.shields.io/badge/SCOOD%20dataset-google%20drive-f39f37)](https://drive.google.com/file/d/1cbLXZ39xnJjxXnDM7g2KODHIjE0Qj4gu/view?usp=sharing)&nbsp;
[![onedrive](https://img.shields.io/badge/SCOOD%20dataset-onedrive-blue)](https://entuedu-my.sharepoint.com/:u:/r/personal/jingkang001_e_ntu_edu_sg/Documents/scood_benchmark.zip?csf=1&web=1&e=vl8nr8)


This repository is the official implementation of the paper:
> **Semantically Coherent Out-of-Distribution Detection**<br>
> Jingkang Yang, Haoqi Wang, Litong Feng, Xiaopeng Yan, Huabin Zheng, Wayne Zhang, Ziwei Liu<br>
> Proceedings of the IEEE International Conference on Computer Vision (**ICCV 2021**)<br>

![udg](assets/udg.png)

## Dependencies
We use `conda` to manage our dependencies, and CUDA 10.1 to run our experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
conda activate scood
```

## SC-OOD Dataset

![scood](assets/benchmark_gif.gif)

The SC-OOD dataset introduced in the paper can be downloaded here.

[![gdrive](https://img.shields.io/badge/SCOOD%20dataset-google%20drive-f39f37)](https://drive.google.com/file/d/1cbLXZ39xnJjxXnDM7g2KODHIjE0Qj4gu/view?usp=sharing)&nbsp;[![onedrive](https://img.shields.io/badge/SCOOD%20dataset-onedrive-blue)](https://entuedu-my.sharepoint.com/:u:/r/personal/jingkang001_e_ntu_edu_sg/Documents/scood_benchmark.zip?csf=1&web=1&e=vl8nr8)

Our codebase accesses the dataset from the root directory in a folder named `data/` by default, i.e.
```
├── ...
├── data
│   ├── images
│   └── imglist
├── scood
├── test.py
├── train.py
├── ...
```


## Training
The entry point for training is the `train.py` script. The hyperparameters for each experiment is specified by a `.yml` configuration file (examples given in [`configs/train/`](configs/train/)).

All experiment artifacts are saved in the specified `args.output_dir` directory.

```bash
python train.py \
    --config configs/train/cifar10_udg.yml \
    --data_dir data \
    --output_dir output/cifar10_udg
```

## Testing
Evaluation for a trained model is performed by the `test.py` script, with its hyperparameters also specified by a `.yml` configuration file (examples given in [`configs/test/`](configs/test/))

Within the configuration file, you can also specify which post-processing OOD method to use (e.g. ODIN or Energy-based OOD detector (EBO)).

The evaluation results are saved in a `.csv` file as specified.

```bash
python test.py \
    --config configs/test/cifar10.yml \
    --checkpoint output/cifar10_udg/best.ckpt \
    --data_dir data \
    --csv_path output/cifar10_udg/results.csv
```

## Results

We report the mean ± std results from the current codebase as follows, which match the performance reported in our original paper.

### CIFAR-10 (+ Tiny-ImageNet) Results

You can run the following script (specifying the data and output directories) which perform training + testing for our main experimental results:

**CIFAR-10, UDG**
```bash
bash scripts/cifar10_udg.sh data_dir output_dir
```

**CIFAR-10 (+ Tiny-ImageNet), ResNet18**

| Metrics      |         ODIN |          EBO |           OE |       UDG (ours) |
| :----------- | -----------: | -----------: | -----------: | ---------------: |
| FPR95 ↓      | 50.76 ± 3.39 | 50.70 ± 2.86 | 54.99 ± 4.06 | **39.94** ± 3.77 |
| AUROC ↑      | 82.11 ± 0.24 | 83.99 ± 1.05 | 87.48 ± 0.61 | **93.27** ± 0.64 |
| AUPR In ↑    | 73.07 ± 0.40 | 76.84 ± 1.56 | 85.75 ± 1.70 | **93.36** ± 0.56 |
| AUPR Out ↑   | 85.06 ± 0.29 | 85.44 ± 0.73 | 86.95 ± 0.28 | **91.21** ± 1.23 |
| CCR@FPRe-4 ↑ |  0.30 ± 0.04 |  0.26 ± 0.09 |  7.09 ± 0.48 | **16.36** ± 4.33 |
| CCR@FPRe-3 ↑ |  1.22 ± 0.28 |  1.46 ± 0.18 | 13.69 ± 0.78 | **32.99** ± 4.16 |
| CCR@FPRe-2 ↑ |  6.13 ± 0.72 |  8.17 ± 0.96 | 29.60 ± 5.31 | **59.14** ± 2.60 |
| CCR@FPRe-1 ↑ | 39.61 ± 0.72 | 47.57 ± 3.33 | 64.33 ± 3.44 | **81.04** ± 1.46 |

**CIFAR-10 (+ Tiny-ImageNet), DenseNet**

| Metrics      |         ODIN |          EBO |           OE |       UDG (ours) |
| :----------- | -----------: | -----------: | -----------: | ---------------: |
| FPR95 ↓      | 51.75 ± 4.22 | 51.11 ± 3.67 | 63.83 ± 8.73 | **43.29** ± 3.37 |
| AUROC ↑      | 86.68 ± 1.74 | 86.56 ± 1.37 | 83.59 ± 3.14 |  **91.8** ± 0.65 |
| AUPR In ↑    | 83.35 ± 2.36 | 84.05 ± 1.75 | 81.78 ± 3.16 | **91.12** ± 0.83 |
| AUPR Out ↑   |  87.1 ± 1.53 | 86.19 ± 1.26 | 82.21 ± 3.51 | **90.73** ± 0.65 |
| CCR@FPRe-4 ↑ |  1.53 ± 0.81 |  2.08 ± 1.07 |  2.57 ± 0.83 |  **8.63** ± 1.86 |
| CCR@FPRe-3 ↑ |  5.33 ± 1.35 |  6.98 ± 1.46 |  7.46 ± 1.66 | **19.95** ± 1.95 |
| CCR@FPRe-2 ↑ | 20.35 ± 3.57 | 23.13 ± 2.92 |  21.97 ± 3.6 | **45.93** ± 3.33 |
| CCR@FPRe-1 ↑ | 60.36 ± 4.47 | 60.01 ± 3.06 | 56.67 ± 5.53 | **76.53** ± 1.23 |

**CIFAR-10 (+ Tiny-ImageNet), WideResNet**

| Metrics      |          ODIN |           EBO |           OE |       UDG (ours) |
| :----------- | ------------: | ------------: | -----------: | ---------------: |
| FPR95 ↓      |  45.04 ± 10.5 |  38.99 ± 2.71 | 43.85 ± 2.68 | **34.11** ± 1.77 |
| AUROC ↑      |  84.81 ± 6.84 |  89.94 ± 2.77 | 91.02 ± 0.54 |  **94.25** ± 0.2 |
| AUPR In ↑    |  77.12 ± 11.7 |  85.39 ± 5.73 |  89.86 ± 0.7 | **93.93** ± 0.12 |
| AUPR Out ↑   |  87.65 ± 4.48 |  90.21 ± 1.81 | 90.11 ± 0.73 | **93.39** ± 0.29 |
| CCR@FPRe-4 ↑ |   2.86 ± 3.84 |   3.88 ± 5.09 |  9.58 ± 1.15 |   **13.8** ± 0.7 |
| CCR@FPRe-3 ↑ |  8.27 ± 10.77 | 10.05 ± 12.32 |  18.67 ± 1.7 | **29.26** ± 1.82 |
| CCR@FPRe-2 ↑ | 19.56 ± 21.85 | 23.58 ± 20.67 | 39.35 ± 2.66 |  **56.9** ± 1.73 |
| CCR@FPRe-1 ↑ | 49.13 ± 24.56 | 67.91 ± 10.61 |  74.7 ± 1.54 |  **83.88** ± 0.2 |

### CIFAR-100 (+ Tiny-ImageNet) Results

You can run the following script (specifying the data and output directories) which perform training + testing for our main experimental results:

**CIFAR-100, UDG**
```bash
bash scripts/cifar100_udg.sh data_dir output_dir
```

**CIFAR-100 (+ Tiny-ImageNet), ResNet18**

| Metrics      |         ODIN |          EBO |           OE |   UDG (ours) |
| :----------- | -----------: | -----------: | -----------: | -----------: |
| FPR95 ↓      | 79.87 ± 0.68 | 78.93 ± 1.39 | 81.53 ± 0.86 | 81.35 ± 0.42 |
| AUROC ↑      | 78.73 ± 0.28 |  80.1 ± 0.46 | 78.67 ± 0.46 | 75.52 ± 0.87 |
| AUPR In ↑    | 79.22 ± 0.28 | 81.49 ± 0.39 | 80.84 ± 0.33 | 74.49 ± 1.89 |
| AUPR Out ↑   | 73.37 ± 0.49 | 73.72 ± 0.44 | 71.75 ± 0.52 | 71.25 ± 0.57 |
| CCR@FPRe-4 ↑ |  1.64 ± 0.51 |   2.55 ± 0.5 |  4.65 ± 0.55 |  1.22 ± 0.39 |
| CCR@FPRe-3 ↑ |   5.91 ± 0.6 |  7.71 ± 1.02 | 11.07 ± 0.43 |  4.58 ± 0.68 |
| CCR@FPRe-2 ↑ | 18.74 ± 0.87 |  22.58 ± 0.8 | 23.26 ± 0.33 | 14.89 ± 1.36 |
| CCR@FPRe-1 ↑ | 46.92 ± 0.15 |  50.2 ± 0.62 | 46.73 ± 0.73 | 39.94 ± 1.68 |

**CIFAR-100 (+ Tiny-ImageNet), DenseNet**

| Metrics      |         ODIN |          EBO |           OE |   UDG (ours) |
| :----------- | -----------: | -----------: | -----------: | -----------: |
| FPR95 ↓      | 83.68 ± 0.57 | 82.18 ± 1.23 | 86.71 ± 2.25 |  80.67 ± 2.6 |
| AUROC ↑      | 73.74 ± 0.84 |  76.9 ± 0.89 | 70.74 ± 2.95 | 75.54 ± 1.69 |
| AUPR In ↑    | 73.06 ± 1.09 | 77.45 ± 1.16 |  70.74 ± 3.0 | 75.65 ± 2.13 |
| AUPR Out ↑   |  69.2 ± 0.65 |  70.8 ± 0.78 | 66.33 ± 2.63 | 70.99 ± 1.62 |
| CCR@FPRe-4 ↑ |  0.55 ± 0.06 |  1.33 ± 0.53 |  1.28 ± 0.33 |   1.68 ± 0.3 |
| CCR@FPRe-3 ↑ |  2.94 ± 0.16 |  4.88 ± 0.82 |  3.81 ± 0.68 |  5.89 ± 1.43 |
| CCR@FPRe-2 ↑ |  11.12 ± 1.1 | 15.53 ± 1.36 | 11.29 ± 1.91 |  16.41 ± 1.8 |
| CCR@FPRe-1 ↑ | 35.98 ± 1.37 | 42.44 ± 1.33 | 31.71 ± 2.73 | 40.28 ± 2.37 |


**CIFAR-100 (+ Tiny-ImageNet), WideResNet**

 | Metrics      |         ODIN |          EBO |           OE |   UDG (ours) |
 | :----------- | -----------: | -----------: | -----------: | -----------: |
 | FPR95 ↓      | 79.59 ± 1.36 | 78.86 ± 1.70 | 80.08 ± 2.80 | 76.03 ± 2.82 |
 | AUROC ↑      | 77.45 ± 0.77 | 80.13 ± 0.56 | 79.24 ± 2.40 | 79.78 ± 1.41 |
 | AUPR In ↑    | 75.25 ± 1.20 | 80.18 ± 0.57 | 80.24 ± 3.03 | 79.96 ± 2.02 |
 | AUPR Out ↑   | 73.2  ± 0.77 | 73.71 ± 0.58 | 73.14 ± 2.19 | 74.77 ± 1.21 |
 | CCR@FPRe-4 ↑ | 0.43  ± 0.21 | 0.58  ± 0.25 | 2.39  ± 0.74 | 1.47  ± 1.08 |
 | CCR@FPRe-3 ↑ | 2.31  ± 0.60 | 3.46  ± 0.80 | 7.97  ± 1.47 | 5.43  ± 2.09 |
 | CCR@FPRe-2 ↑ | 11.01 ± 1.29 | 17.55 ± 1.24 | 21.97 ± 2.92 | 18.88 ± 3.53 |
 | CCR@FPRe-1 ↑ | 43.2  ± 1.80 | 51.54 ± 0.65 | 49.36 ± 3.98 | 48.95 ± 1.91 |

**Note:**
The work was originally built on the company's own deep learning framework, based on which we report all the results in the paper.
We extracted all related code and built this standalone version for release, and checked that most of the results can be reproduced. 
We noticed that CIFAR-10 can easily match the paper results, but CIFAR-100 benchmark might have a few differences, perhaps due to some minor difference in framework modules and some randomness. 
We are currently enhancing our codebase and exploring udg on large-scale datasets.

## License and Acknowledgements
This project is open-sourced under the MIT license.

The codebase is refactored by Ang Yi Zhe, and maintained by Jingkang Yang and Ang Yi Zhe.

## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@InProceedings{yang2021scood,
    author = {Yang, Jingkang and Wang, Haoqi and Feng, Litong and Yan, Xiaopeng and Zheng, Huabin and Zhang, Wayne and Liu, Ziwei},
    title = {Semantically Coherent Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
    year = {2021}
}
```
