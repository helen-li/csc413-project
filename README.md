# Can Modular Architectures Forget the Noise?
___
## Introduction

Building off the work published in the paper **[Can Modular Architectures Forget the Noise?](https://arxiv.org/abs/2206.02713)**, this project explores the effect of Gaussian noise on various modular architectures. In particular, we sought to analyze the performance of MLP-based, RNN-based, and MHA-based systems in response to the addition of noise in the last step of data generation.

The main idea behind testing this is that data in the real-world rarely follows a fixed distribution according to specific data-generating rules, as implemented in the original paper. Rather, while data may follow trends and patterns that correspond to deterministic data distributions, there is often randomness that must be accounted for when analyzing differences between various model architectures. 

To account for this fact, we decided to manually add Gaussian noise in the last stage of the data generation process. In order to get a sense of how various levels of noise impact model performance, we added Gaussian noise with 6 different standard deviation values: 0 (no noise), 0.1, 0.25, 0.5, 1.0, and 2.0. For each standard deviation, we trained models over five different numbers of rules, similar to the paper: 2, 4, 8, 16, and 32. 

For each of the three model architecture, we tested various levels of model modularity, to gauge how the addition of noise impacts modular vs. monolithic architectures. As defined in the original paper, we conducted experiments on Monolithic, Modular, Modular-op, and GT-Modular systems.

---

## Running the code

In the root directory, there are three folders for each of the model architectures: ```MLP```, ```RNN```, and ```MHA```. In each folder, there are subdirectories for ```Regression``` and ```Classification```. To train a specific model architecture for either regression or classification, go to the respective directory and run one of the ```train``` scripts.

Note that there are six such scripts in each folder for each of the 6 noise levels:

- ```train-no-noise.sh```
- ```train-std-0.1.sh```
- ```train-std-0.25.sh```
- ```train-std-0.5.sh```
- ```train-std-1.0.sh```
- ```train-std-2.0.sh```

Each training script will run each of the four modular systems — Monolithic, Modular, Modular-op, and GT-Modular — with the five different numbers of rules (2, 4, 8, 16, and 32).

After training is complete, you will have to evaluate your models. In the same subdirectory, you will find an ```evaluate.sh``` script. Running this will produce metrics that are saved in a new ```Logistics``` directory. 

From here, you can use ```plot_main``` and ```plot.py``` to generate the main plots shown in the paper and more detailed plots for each specific architecture, respectively. ```plot_main``` works with the existing Logistics folders in this repository and saves all figures into a new ```Main_Plots``` directory. ```plot.py``` works with six ```Logistics``` directories (one for each architecture and each task) to generate more specific plots for each noise level, which were omitted in the paper in the interest of space. 