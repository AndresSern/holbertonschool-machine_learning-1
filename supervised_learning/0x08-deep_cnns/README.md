4. ResNet-50 
the first block is the projection block and the rest are identity blocks. So for ‘conv2_x’, the first block is a projection block and the other two repetitions are identity blocks.
The reason the first block is projection is to make sure, that the strides and the number of filters of the input from skip connection and the actual output of the block are the same. In the projection block, the first convolutional layer has strides=2. This means that if the input is an image, the size of the image after this layer will decrease. But the input in skip connection still has the previous image size. Adding two images of different sizes is not possible. So the skip connection also has a convolutional layer with stride=2. This makes sure that the image sizes are now the same.

5-
