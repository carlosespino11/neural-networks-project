"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of some basic components in neural network.

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
import scipy
floatX = theano.config.floatX
#from theano.tensor.signal import downsample

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, p=0.5):
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(output,p)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)
        
        # parameters of the model
        self.params = [self.W, self.b]
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),st_conv=(1,1),st_pool=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            subsample=st_conv
            
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True,
            st=st_pool
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
def drop(input, p=0.7): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """            
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask
class DropoutLayer(object):
    def __init__(self,input, is_train,image_shape,n_in, n_out, p=0.5):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        if(is_train==0):
            image_shape=(8,image_shape[1],image_shape[2],image_shape[3])
        self.input = input
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(input,p).reshape(image_shape)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0),train_output, train_output)
class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, border_mode=1):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode = border_mode
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.ypred = T.tanh(conv_out + self.b.dimshuffle(8, 3, 32,32))
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
    def mean_squared_error(self, y):
        return T.mean(T.pow(self.output - y,2))
class Unpooling_2D(object):
    def __init__(self,input, ds=(2, 2), ignore_border=True):
        self.input = input
        self.ds = ds
        self.ignore_border = ignore_border
        X = self.input
        s1 = self.ds[0]
        s2 = self.ds[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        self.output =output

def adam(params, cost, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, grad_clip=None):
    """Adam algorithm proposed was proposed in Adam: A Method for Stochastic
    Optimization.
    This code was found in:
    https://github.com/EderSantana/top/blob/master/top/update_rules.py
    :param params: list of :class:theano.shared variables to be optimized
    :param cost: cost function that should be minimized in the optimization
    :param float lr: learning rate
    :param float b1: ToDo: WRITEME
    :param float b2: ToDo: WRITEME
    :param float e: ToDO: WRITEME
    """
    updates = []
    grads = T.grad(cost, params)
    zero = np.zeros(1).astype(floatX)[0]
    i = theano.shared(zero)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        if grad_clip is not None:
            gnorm = T.sqrt(T.sqr(g).sum())
            ggrad = T.switch(T.ge(gnorm,grad_clip),
                             grad_clip*g/gnorm, g)
        else:
            ggrad = g
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * ggrad) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(ggrad)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def noise_one_image(input_im,noise):
    if(numpy.random.uniform(low=0.0, high=1.0))>=0.5:
        noise=numpy.random.normal(loc=0.0, scale=1.0*noise, size=numpy.shape(input_im))
    else:
        noise=numpy.random.uniform(low=-1.0*noise, high=1.0*noise,size=numpy.shape(input_im))
    output=input_im+noise
    return output.transpose(2,0,1).flatten()
def noise_injection(input_im,input_y,factor=1,function=noise_one_image,noise=0.01,theano_shared=True,all_data=True):
    l=len(input_im)
    imput_data_aux=input_im
    it_ix=[]
    if(all_data):
        data_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
        imput_data_aux=input_im[data_ix]
        it_ix=range(0,len(imput_data_aux))
    else:
        it_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
    
    for i in it_ix:
        input_aux=numpy.reshape(imput_data_aux[i],(3,32,32)).transpose(1,2,0)
        imput_data_aux[i]=function(input_aux,noise)
    if(all_data):
        input_im=numpy.concatenate((input_im,imput_data_aux),axis=0)
        result_y=input_y[data_ix]
        input_y=numpy.concatenate((input_y,result_y),axis=0)
    else:
        input_im=imput_data_aux
    if theano_shared:
        shared_x= theano.shared(numpy.asarray(input_im,
                                           dtype=theano.config.floatX),
                             borrow=True)
        shared_y=theano.shared(numpy.asarray(input_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
        return shared_x,T.cast(shared_y, 'int32')
    else:
        return input_im,input_y
def flip_one_image(input_im):
    output=numpy.fliplr(input_im)
    return output.transpose(2,0,1).flatten()
def flip_image(input_im,input_y,factor=1,function=flip_one_image,theano_shared=True,all_data=True):
    l=len(input_im)
    imput_data_aux=input_im
    it_ix=[]
    if(all_data):
        data_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
        imput_data_aux=input_im[data_ix]
        it_ix=range(0,len(imput_data_aux))
    else:
        it_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
    
    for i in it_ix:
        input_aux=numpy.reshape(input_im[i],(3,32,32)).transpose(1,2,0)
        imput_data_aux[i]=function(input_aux)
    if(all_data):
        input_im=numpy.concatenate((input_im,imput_data_aux),axis=0)
        result_y=input_y[data_ix]
        input_y=numpy.concatenate((input_y,result_y),axis=0)
    else:
        input_im=imput_data_aux
    if theano_shared:
        shared_x= theano.shared(numpy.asarray(input_im,
                                           dtype=theano.config.floatX),
                             borrow=True)
        shared_y=theano.shared(numpy.asarray(input_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
        return shared_x,T.cast(shared_y, 'int32')
    else:
        return input_im,input_y
def rotate_one_image(angle,input_im):
    output=scipy.ndimage.interpolation.rotate(input_im, angle,reshape=False,
                                              order=3, mode='constant', cval=0.0, prefilter=True)
    return output.transpose(2,0,1).flatten().reshape((1,-1))
def rotate_image(input_im,input_y,max_angle, factor=1,function=rotate_one_image,theano_shared=True,all_data=True):
    l=len(input_im)
    imput_data_aux=input_im
    it_ix=[]
    if(all_data):
        data_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
        imput_data_aux=input_im[data_ix]
        it_ix=range(0,len(imput_data_aux))
    else:
        it_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
    
    for i in it_ix:
        input_aux=numpy.reshape(imput_data_aux[i],(3,32,32)).transpose(1,2,0)
        angle=numpy.random.uniform(low=-max_angle, high=max_angle)
        imput_data_aux[i]=function(angle,input_aux)
    if(all_data):
        input_im=numpy.concatenate((input_im,imput_data_aux),axis=0)
        result_y=input_y[data_ix]
        input_y=numpy.concatenate((input_y,result_y),axis=0)
    else:
        input_im=imput_data_aux
    if theano_shared:
        shared_x= theano.shared(numpy.asarray(input_im,
                                           dtype=theano.config.floatX),
                             borrow=True)
        shared_y=theano.shared(numpy.asarray(input_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
        return shared_x,T.cast(shared_y, 'int32')
    else:
        return input_im,input_y
def translate_one_image(n,m,input_im):
    output=scipy.ndimage.interpolation.shift(input_im, [m,n,0],order=3, mode='constant', cval=0.0, prefilter=True)
    return output.transpose(2,0,1).flatten()
def translate_image(input_im,input_y,max_m, max_n, factor=1,function=translate_one_image,theano_shared=True,all_data=True):
    l=len(input_im)
    imput_data_aux=input_im
    it_ix=[]
    if(all_data):
        data_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
        imput_data_aux=input_im[data_ix]
        it_ix=range(0,len(imput_data_aux))
    else:
        it_ix=numpy.random.choice(range(0,l), ((l)*factor), replace=False)
    
    for i in it_ix:
        n=numpy.around(max_n*numpy.random.uniform(low=0.0, high=1.0))
        m=numpy.around(max_m*numpy.random.uniform(low=0.0, high=1.0))
        input_aux=numpy.reshape(input_im[i],(3,32,32)).transpose(1,2,0)
        imput_data_aux[i]=result_val=function(n,m,input_aux)
    if(all_data):
        input_im=numpy.concatenate((input_im,imput_data_aux),axis=0)
        result_y=input_y[data_ix]
        input_y=numpy.concatenate((input_y,result_y),axis=0)
    else:
        input_im=imput_data_aux
    if theano_shared:
        shared_x= theano.shared(numpy.asarray(input_im,
                                           dtype=theano.config.floatX),
                             borrow=True)
        shared_y=theano.shared(numpy.asarray(input_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
        return shared_x,T.cast(shared_y, 'int32')
    else:
        return input_im,input_y
def generate_data(x2,y2,factor=0.1,function_arg="noise,flip,rotate,translate"):
    x2=numpy.copy(x2)
    y2=numpy.copy(y2)
    func=function_arg.split(",")
    if("flip" in func):
        x2,y2=flip_image(x2,y2,factor=0.1,theano_shared=False,all_data=False)
    if("noise" in func):
        x2,y2=noise_injection(x2,y2,factor=0.01,theano_shared=False,all_data=False)
    if("rotate" in func):
        x2,y2=rotate_image(x2,y2,15,factor=0.1,theano_shared=False,all_data=False)
    if("translate" in func):
        x2,y2=translate_image(x2,y2,3,3,factor=0.1,theano_shared=False,all_data=False)
    return x2

def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True,patience = 10000,factor=0.1,data_x=[],
            data_y=[],batch_size=500,function_arg="noise,flip,rotate,translate"):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            x_aug=data_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size]    
            if(factor>0):
                x_aug=generate_data(data_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                                data_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],factor,function_arg)
            cost_ij = train_model(minibatch_index,x_aug)
             

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
def train_2nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True,patience = 10000):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation mse %f ' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean([test_losses[test_ix][0] for test_ix in range(len(test_losses))])

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f ') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f obtained at iteration %i, '
          'with test performance %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    img1=[test_losses[test_ix][1] for test_ix in range(len(test_losses))]
    img2=[test_losses[test_ix][2] for test_ix in range(len(test_losses))]
    img3=[test_losses[test_ix][3] for test_ix in range(len(test_losses))]
    return img1,img2,img3