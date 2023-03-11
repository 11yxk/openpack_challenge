# ST-GCN

## Notebooks

### [ST-GCN_Example_01.ipynb](./notebooks/ST-GCN_Example_01.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/st-gcn/notebooks/ST-GCN_Example_01.ipynb)

In this notebook, you can train and test the ST-GCN with `openpack_torch` package.
The basic ST-GCN model is designed for classification task, so we modified it for segmentation task by setting all strides = 1, and replacing the output layer.

## Script

```bash
# Training
$ python main.py mode=train debug=false

# Test
$ python main.py mode=test debug=false

# Make submission zip file
$ python main.py mode=submission debug=false
```

## Reference

- [Paper - AAAI2018 - Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- [GitHub - machine-perception-robotics-group/MPRGDeepLearningLectureNotebook](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook)
  - [Colab - 03_action_recognition_ST_GCN.ipynb](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb)
- [GitHub - open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton/blob/master/doc/START_RECOGNITION.md)
