## Hubness Reduction with Dual Bank Sinkhorn Normalization for Cross-Modal Retrieval (DBSN)<br><sub>Official PyTorch implementation of the ACMMM 2025 paper</sub>

**Hubness Reduction with Dual Bank Sinkhorn Normalization for Cross-Modal Retrieval**<br>
Zhengxin Pan, Haishuai Wang, Fangyu Wu, Peng Zhang, Jiajun Bu
<br>https://arxiv.org/abs/2507.00364<br>

## Requirements and Installation
We recommended the following dependencies.

* Python 3.7
* librosa
* [PyTorch](http://pytorch.org/) 1.11.0
* [Transformers](https://github.com/huggingface/transformers) (4.18.0)

## Results
### Text-Image Retrieval
please refer to [t2i](./Image_Retrieval/readme.md).

### Text-Video Retrieval
please refer to [t2v](./Video_Retrieval/readme.md).

### Text-Audio Retrieval
please refer to [t2a](./Audio_Retrieval/readme.md).

### Image-Image Retrieval
please refer to [i2i](./hyp_metric/README.md).

### Text-Medical Retrieval
please refer to [t2m](./Medical_Retrieval/readme.md).

### Zero-shot Image Classification
please refer to [ic](./Image_Classification/readme.md).

### Visualizations
please refer to [vis](./visualization.ipynb).

## Reference
If you found this code useful, please cite the following paper:
```
@inproceedings{pan2025dbsn,
    title={Hubness Reduction with Dual Bank Sinkhorn Normalization for Cross-Modal},
    author={Zhengxin Pan, Haishuai Wang, Fangyu Wu, Peng Zhang, Jiajun Bu},
    booktitle={Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland},
    year={2025}
} 
```

