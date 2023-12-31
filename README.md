
# RACK
This is an official PyTorch implementation of RACK, as outlined below: 

Xiaoli Liu, Jianqin Yin, Di Guo, and Huaping Liu. Rich Action-semantic Consistent Knowledge for Early Action Prediction[J]. IEEE Transactions on Image Processing (TIP), 2023, doi:10.1109/TIP.2023.3345737.

https://arxiv.org/abs/2201.09169

## Setup
Required python libraries: pytorch (>=1.0)  + numpy.

Tested in ubuntu +  GTX 3080Ti with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
UCF101, HMDB51, Something-Something V2.

We will make our preprocessed features available on Baidu Cloud, including features of different backbones/datasets. Please reffer to: https://pan.baidu.com/s/1gm6ZjQvG4hUWCKgJ8S3xkA （password：RACK）



## Training/Testing
train/testing of RACK-RGB or RACK-flow
```shell
python train.py --trainfile  "<path of training data>" --tr_labelfile "<path of training label>" --testfile "<path of test data>" --te_labelfile  ="<path of test label>" --save_path "<path to save model>"  --save_results "<path of save testing results>" 
```
To train and test on different datasets/modalities, it is necessary to substitute the corresponding features.

# Fusion
```shell
python two_stream_fusion.py --path_rgb_logits  "<logits of RACK-RGB>" --path_flow_logits "<logits of RACK-flow>" --telabel "<test label>" --segments "<Number of all progress levels>"
```

# Failed cases
We will present and analyze some failed cases in "Some in-depth analysis of the failed cases on HMDB51 in Fig. 5.pdf".


# Results
More results with different combinations are available at "our results with different features.xlsx".


## Citation
If you use this code for your research, please consider citing:
```latex
@article{liu2023RACK,
  title={Rich Action-semantic Consistent Knowledge for Early Action Prediction},
  author={Liu, Xiaoli and Yin, Jianqin and Di, Guo and Liu, Huaping},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  doi={10.1109/TIP.2023.3345737};
  publisher={IEEE}
}
```

## Contact
If you have any questions, please contact us via email: liuxiaoli134@bupt.edu.cn.
