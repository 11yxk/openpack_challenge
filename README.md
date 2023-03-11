# openpack_challenge
Pytorch implementation of paper '' A Spatial-Temporal Graph Convolutional Networks-based Approach for the OpenPack Challenge 2022''. 
We won the 3rd place in the [OpenPack challenge 2022](https://open-pack.github.io/challenge2022).


# Data Preparation
Please download data from [OpenPack dataset](https://open-pack.github.io/release/v0-3-1).

# Training & Testing

### Training

```
# Example: training network on NTU RGB+D 60 cross subject
python main.py --config ./config/nturgbd-cross-subject/default.yaml
```
### Testing

```
# Example: testing the joint modality of nturgbd-cross-subject dataset using second scheme
python main.py --config ./config/nturgbd-cross-subject/default.yaml --phase test --save-score True --weights weight/CTR-GCN-Scheme2/ntu60/xsub/CS_joint.pt --model model.dynamic_ctrgcn_scheme2_test.Model
```

- To ensemble the results of different modalities, run:
```
# Example: ensemble four modalities on NTU RGB+D 60 cross subject
python ensemble.py --datasets ntu/xsub --joint-dir work_dir/ntu/xsub/ctrgcn --bone-dir work_dir/ntu/xsub/ctrgcn_bone --joint-motion-dir work_dir/ntu/xsub/ctrgcn_motion --bone-motion-dir work_dir/ntu/xsub/ctrgcn_bone_motion
```

### Pretrained Models

- Pretrained Models are avaliable at https://drive.google.com/file/d/16tBLRYZszba8-nK1GVfQ-ESHVuM-c2yG/view?usp=share_link.


## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

Thanks original authors for their work!
