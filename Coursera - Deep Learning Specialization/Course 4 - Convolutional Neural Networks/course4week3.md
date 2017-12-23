#Week 3

## Detection Algorithms

*	Detection of object in an image patch is a straightforward CNN task
*	Encode for each patch a size 8 vector

	Type | Value
	--- | ---
	Is object present [p<sub>c</sub>] | 0/1
	b<sub>x</sub> | ~
	b<sub>y</sub> | ~
	b<sub>h</sub> | ~
	b<sub>w</sub> | ~
	c<sub>1</sub> | 0/1
	c<sub>2</sub> | 0/1
	... | ...

*	To cover the entire image, simply slide the window over the entire input image -> Computationally expensive because of iterative implementation
*	Instead, the FC layers of the CNN can be represented as a convolution with large filter. E.g.
>	FC: 5x5x128 -> 400 == CONV: filter_sz = 5x5 num_filters = 400
*	This allows the [entire image to be processed by a single CNN](https://arxiv.org/abs/1312.6229 "OverFeat") which is more efficiently implemented
*	To get accurate bounding boxes quickly, [YOLO](https://arxiv.org/abs/1612.08242 "You Look Only Once") processes the input image, as with OverFeat, to generate an output volume with 8 channels (encoded as above) and then marks bounding boxes by thresholding on p<sub>c</sub>. 
	*	To generate the target output volume, object is assigned to a cell if its centre lies in it.
	*	b<sub>x</sub> and b<sub>y</sub> are defined relative to the cell corner (top-left)
	*	b<sub>h</sub> and b<sub>w</sub> are defined as fractions of cell dimensions
*	**Evaluation of detected bounding box:** Metric of Intersection over Union (IoU) is used
>	IoU = Intersection of prediction and groundtruth / Union of prediction and groundtruth
>	IoU >= 0.5 is considered a good detection
*	**Non-maximal suppresion:** Used to avoid cluttered multiple bounding boxes for the same object
	*	Discard all BB with low p<sub>c</sub>
	*	While there are unprocessed BBs, take box with highest p<sub>c</sub>, output it as a prediction and discard all bounding boxes with a high overlap (in terms of IoU) with this BB
*	**Anchor boxes:** Used to handle conditions where multiple object centres fall in the same cell and/or allow CNN nodes to specialise to specific box types
	*	Basic change is in how output volume is represented. There are now `1+4+num_classes` values per anchor box per cell i.e. output volume is `num_cell_h x num_cell_w x (1+4+num_classes)*num_anchor_box`
	*	Object belonging to a cell is assigned to the anchor box having highest overlap (IoU) with object bounding box
	*	*Note: Non-max suppression is performed per class i.e. discard high overlap boxes predicting the same class as the box under consideration*

## Region proposals
*	[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524 "R-CNN")
	*	Propose regions by segmentation
	*	Classify each proposed region
*	[Fast R-CNN](https://arxiv.org/abs/1504.08083 "Fast R-CNN")
	*	Convolutional implementation of sliding windows
*	[Faster R-CNN](https://arxiv.org/abs/1506.01497 "Faster R-CNN")
	*	Convolutional implementation of region proposal
	
## Notes from programming assignment
*	`tf.boolean_mask`
```python
boolean_mask(
    tensor,
    mask,
    name='boolean_mask'
)
```
>	In general, `0 < dim(mask) = K <= dim(tensor)`, and mask's shape must match the first K dimensions of tensor's shape. We then have: `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]` where `(i1,...,iK)` is the ith `True` entry of mask (row-major order).	

*	`tf.image.non_max_suppression`
```python
non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    name=None
)
```
>	The output of this operation is a set of integers indexing into the input collection of bounding boxes representing the selected boxes. The bounding box coordinates corresponding to the selected indices can then be obtained using the `tf.gather` operation. For example: selected_indices = tf.image.non_max_suppression( boxes, scores, max_output_size, iou_threshold) selected_boxes = tf.gather(boxes, selected_indices)