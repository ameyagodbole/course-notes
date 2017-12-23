# Week 4

## Face Recognition
*	**Verification** (match one face to another i.e. one-to-one task) vs **Recognition** (match face to database i.e. one-to-many task)
*	**One shot learning** (only one sample of each class is available)
*	Direct CNN with 1 output node per person in the database is not feasible because there is not enough data to train the CNN `(num_samples = num_persons)` and also because any addition to the database (new member) will force retraining of entire model
*	Instead use a **similarity** measure: outputs degree of similarity between 2 images
*	**Triplet loss:** `max( d(A,P) - d(A,N) + margin, 0 )`
	A	|	anchor	|	Image under consideration
	P	|	positive	|	Image of the same class
	N	|	negative	|	Image of a different class
*	For faster convergence, the triplets need to be selected efficiently as detailed in [FaceNet](https://arxiv.org/abs/1503.03832)
*	*Note:* Database encodings can be precomputed and stored on disk to decrease run time

## Neural Style Transfer
Generate an image (G) that paints content image (C) in the format/pattern of style image (S)

*	J(C,S,G) = alpha * J<sub>content</sub>(C,G) + beta * J<sub>style</sub>(S,G)
*	**Content cost**
	1.	Use pretrained ConvNet
	2.	Let a<sup>[l]G</sup> and a<sup>[l]C</sup> be activations of layer l of ConvNet on G and C respectively
	3.	J<sup>[l]</sup><sub>content</sub>(C,G) = 0.5 * || a<sup>[l]G</sup> - a<sup>[l]C</sup>||<sup>2</sup>
*	**Style cost**
	Correlation between channels of a layer gives sense of how often the features described by the channel occur together
	1.	Let a<sup>[l]</sup><sub>i,j,k</sub> be activation at position (i,j,k) of layer l
	2.	*Style matrix* G<sup>[l]</sup> is (n<sub>c</sub>,n<sub>c</sub>)-matrix such that
		G<sup>[l]</sup><sub>kk'</sub> = sum<sub>i</sub>(	sum<sub>j</sub>(	a<sup>[l]</sup><sub>i,j,k</sub>	*	a<sup>[l]</sup><sub>i,j,k'</sub>	)	)
	3.	J<sup>[l]</sup><sub>style</sub>(S,G) = || G<sup>[l]S</sup> - G<sup>[l]G</sup>||<sup>2</sup><sub>F</sub> / (2 * n<sup>[l]</sup><sub>H</sub> * n<sup>[l]</sup><sub>W</sub> * n<sup>[l]</sup><sub>C</sub> )
*	In computing total style and content loss, a weighted sum of respective layerwise losses is used