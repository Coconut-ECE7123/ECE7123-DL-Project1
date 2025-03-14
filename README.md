# NYU ECE-GY7123 MiniProject1 - Training ResNet on CIFAR-10
## Project Info
**Team Name**: Coconut <br>
**Team Member**: Ziyan Zhao; Haotian Xu; Tianqi Xia <br>
<br>
`checkpoint`: model path file (model weights; optimizer state; training record; metrics) 
>`leaky_SOTA.pth`: best model we trained in this project
<br>

`data`: dataset file
>`cifar-10-batches-py`: CIFAR-10 dataset train + test <br>
>`cifar_test_nolabel.pkl`: Kaggle test set https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1/data 
<br>

`models`: model structure description python codes file (pytorch)
>`_init_.py` <br>
>`resnet.py`: original ResNet model code <br>
>`resnet_enhanced.py`: customized ResNet model code
<br>

`main.py`: model training and validation code on CIFAR-10 dataset <br>
`read_test.ipynb`: code for model inference on Kaggle competition dataset <br>
`requirements.txt`: librabries required for env. <br>
`utils.py`: some helper functions for PyTorch (get_mean_and_std; msr_init; progress_bar) <br>


