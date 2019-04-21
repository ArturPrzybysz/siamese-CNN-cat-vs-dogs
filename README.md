#### Cats vs dogs using Siamese CNN with Triplet Loss
I treat this project as an introduction to Siamese Networks 
before facing a more sophisticated problem, which I will describe in my
Bachelor thesis.

#### Dataset
Dataset consists of 12500 cat and 12500 dogs jpg images.  
https://www.kaggle.com/c/dogs-vs-cats/data  

#### Input
TODO: show example input image and describe what are triples (anchor, negative, positive)  

#### Triples construction
TODO: describe, citing paper  

### Evaluation method
Model is evaluated on a validation set after each epoch. 
Validation set consists of a group of random triplets.
I measure distances of ANCHOR-POSITIVE and ANCHOR-NEGATIVE, and count the ratio of positives being closer
than negative to the whole size of validation set.

#### Related papers, articles and repositories
~~https://arxiv.org/pdf/1503.03832.pdf~~  
~~https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d~~  
https://thelonenutblog.wordpress.com/2017/12/14/do-telecom-networks-dreams-of-siamese-memories/  
~~https://github.com/noelcodella/tripletloss-keras-tensorflow/blob/master/tripletloss.py~~  
https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24  

stucking at loss==margin issue  
https://github.com/omoindrot/tensorflow-triplet-loss/issues/6  


#### Results
None so far :)  

#### My config
Python 3.6, tensorflow-gpu 1.8.0, GTX 850M  

#### TODO
online triples generation: https://omoindrot.github.io/triplet-loss  
generators: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly  
consider lossless triplet loss function  
visualisations like https://youtu.be/uWODMSMTIJQ