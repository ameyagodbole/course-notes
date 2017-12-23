#Week 2

##Case Studies

###[LeNet-5](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
Standard traits

*	Height and width decreases with depth
*	Number of channels increases with depth

Unique traits

*	Sigmoid ans tanh instead of ReLU
*	Non-linearity after pooling
*	Different filters look at different channels of filter block (Useful for saving computation)
*	Final layer considers Euclidean Radial Basis Function (RBF) output

### [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
Standard traits

*	Many more parameters
*	ReLU activation

Unique traits

*	2 stream architecture to accomodate for weaker GPUs
*	__Local Response Normalization (LRN) :__ Normalize each position across all channels (Not widely used anymore)

### [VGG nets](https://arxiv.org/abs/1409.1556)
Standard traits

*	Standardized filter sizes and strides so that layers look simplified

### [Resnet](https://arxiv.org/abs/1512.03385)
Unique traits

*	**Skip connections (SC)** across a conv block
	*	Skipped layer has the ability to learn an identity transform -> Deeper network cannot hurt performance
	*	Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients. SC allows better flow of gradient during back propogation
	*	There are two main type of blocks: The identity block and the convolutional block.
	
### [Network in Network](https://arxiv.org/abs/1312.4400)
Unique traits

*	Introduces **1x1 conv**
	*	May help to shrink the number of channels
	*	May be used to increase complexilty with smaller computational cost
	
### [Inception Network](https://arxiv.org/abs/1409.4842)
Unique traits

*	Introduces **inception module**
	*	Tries to do away with filter size as a hyperparameter by incorporating 1x1, 3x3, 5x5 filters and max pooling (in 'same' mode) at same depth
	*	**Bottleneck** layer (e.g. 1x1 before 5x5) to reduce computations
*	**Side branches: ** Hidden layers also enforced to learn different representations for each output class. This method seems to have a regularizing effect

## Practical Advice
1.	Transfer Learning
	*	Use pretrained weights to tranfer knowledge from network trained on larger public dataset to learn on smaller related dataset
2.	Data augmentation
	*	Common techniques
		*	Mirroring
		*	Random cropping
		*	Warping
	*	Other methods
		*	Color shifting (Alexnet used PCA to determine the shift)
3.	Ensembling (Combine outputs of multiple different models)
4.	Combine outputs of **Multiple crops** on test images

## Notes from programming assignment
1.	Displaying model in Keras
```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

Model = SomeKerasModel()

plot_model(Model, to_file='Model.png')
SVG(model_to_dot(Model).create(prog='dot', format='svg'))
```
