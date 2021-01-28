## Description

As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.

Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.

## Task
The Task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With the solution farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

## Dataset
1. [train/test]_images the image files.
2. train.csv
    
    **image_id** : the image file name.

    **label** : the ID code for the disease.
3. sample_submission.csv : A properly formatted sample submission, given the disclosed test set content.

    **image_id** : the image file name.

    **label** :  the predicted ID code for the disease.
4. [train/test]_tfrecords the image files in tfrecord format.

5. labelnumtodiseasemap.json The mapping between each disease code and the real disease name.
## Proposed Solution
###   Model: 

[Visual Transformers](https://github.com/lukemelas/PyTorch-Pretrained-ViT) (ViT) are a straightforward application of the transformer architecture to image classification. Even in computer vision, it seems, attention is all you need.

The ViT architecture works as follows: (1) it considers an image as a 1-dimensional sequence of patches, (2) it prepends a classification token to the sequence, (3) it passes these patches through a transformer encoder (like BERT), (4) it passes the first token of the output of the transformer through a small MLP to obtain the classification logits. ViT is trained on a large-scale dataset (ImageNet-21k) with a huge amount of compute.

!["image"](https://github.com/google-research/vision_transformer/raw/fca1235b4828b10eee120985339ede485afc1ed6/vit_figure.png)

### Layer freezing in transfer learning:
The basic idea is that all models have a function model.children() which returns it’s layers. Within each layer, there are parameters (or weights), which can be obtained using .param() on any children (i.e. layer). Now, every parameter has an attribute called requires_grad which is by default True. True means it will be backpropagrated and hence to freeze a layer you need to set requires_grad to False for all parameters of a layer.inside epoch loop we are freezing first 3 layers in the total N layers of ViT large model. 
!["transfer learning"](https://qph.fs.quoracdn.net/main-qimg-96376d794775a37a272dac2a7a38f29e)

**Cross Validation Strategy**:  &nbsp; stratified K Fold with 5 folds

**Loss**:   &nbsp; [**bi tempered logistic loss**](https://arxiv.org/pdf/1906.03361.pdf)

a temperature into the exponential function and replace the softmax
output layer of neural nets by a high temperature generalization. Similarly, the
logarithm in the log loss one can  use for training is replaced by a low temperature
logarithm. By tuning the two temperatures one can create loss functions that are nonconvex already in the single layer case. When replacing the last layer of the neural
nets by our bi-temperature generalization of logistic loss, the training becomes more
robust to noise. one can visualize the effect of tuning the two temperatures in a simple
setting and show the efficacy of our method on large data sets. Our methodology is
based on Bregman divergences and is superior to a related two-temperature method
using the Tsallis divergence.

### Implementation with code:      [git repo](https://github.com/prakashsellathurai/cassava-leaf-disease-classification)
## Citations
```
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
@inproceedings{amid2019robust,
  title={Robust bi-tempered logistic loss based on bregman divergences},
  author={Amid, Ehsan and Warmuth, Manfred KK and Anil, Rohan and Koren, Tomer},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15013--15022},
  year={2019}
}

```
