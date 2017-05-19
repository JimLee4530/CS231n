import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet(object):
	def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

		self.params = {}
    	self.reg = reg
    	self.dtype = dtype

    	self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    	self.params['b1'] = np.zeros((num_filters,))
    	self.params['W2'] = weight_scale * np.random.randn(num_filters,)
    	self.params['W5'] = weight_scale * np.random.randn(num_filters*input_dim[1]*input_dim[2]/4,hidden_dim)
    	# print self.params['W2'].shape
    	self.params['b5'] = np.zeros((hidden_dim,))
    	self.params['W6'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    	self.params['b5'] = np.zeros((num_classes,))



    def loss(self, X, y=None):
    	W1, b1 = self.params['W1'], self.params['b1']
    	W2, b2 = self.params['W2'], self.params['b2']
    	W3, b3 = self.params['W3'], self.params['b3']
    
    	# pass conv_param to the forward pass for the convolutional layer
    	filter_size = W1.shape[2]
    	conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    	# pass pool_param to the forward pass for the max-pooling layer
    	pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    	scores = None

    	cnn_out, cnn_cache =conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    	ar_out,ar_cache = affine_relu_forward(cnn_out, W2, b2)
    	scores, af_cache = affine_forward(ar_out,W3,b3)


    
    	if y is None:
      		return scores
    
    	loss, grads = 0, {}

    	data_loss,dscores = softmax_loss(scores,y)
    	dar_out,dW3,db3 = affine_backward(dscores,af_cache)
    	dcnn_out,dW2,db2 = affine_relu_backward(dar_out,ar_cache)
    	dx,dW1,db1 = conv_relu_pool_backward(dcnn_out,cnn_cache)
    	# print self.reg
    	grads['W1'] = dW1+self.reg*W1
    	grads['W2'] = dW2+self.reg*W2
    	grads['W3'] = dW3+self.reg*W3
    	grads['b1'] = db1
    	grads['b2'] = db2
    	grads['b3'] = db3

    	reg_loss = 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    	loss = data_loss + reg_loss
    
    	return loss, grads