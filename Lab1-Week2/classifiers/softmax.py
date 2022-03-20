import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    output = X @ W # shape is (N,C) where N is number of training examples, C is number of classes
    exp_output = np.exp(output)
    n, c = np.shape(output)
    
    for i in range(n): #for each training example
      softmaxsum = 0
      dW[:, y[i]] += (np.exp(output[i, y[i]]) / np.sum(np.exp(output[i])) - 1) * X[i]
      for j in range(c): #for each class
        softmaxsum += exp_output[i][j]
        dW[:,j] += (np.exp(output[i,j])/np.sum(np.exp(output[i]))) * X[i]
      exp_output[i] /= softmaxsum
      loss += -np.log(exp_output[i])[y[i]]

    loss = loss/n + 0.5*reg*np.sum(W**2)
    
    dW = dW/n + reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    n = X.shape[0]
    output = X@W
    norm_output = np.exp(output) / np.sum(np.exp(output), axis = 1).reshape(n, 1)
    
    loss = -np.sum(np.log(norm_output[np.arange(n), y]))
    loss = loss/n + 0.5 * reg * np.sum(W * W)

    norm_output[np.arange(n), y] -= 1
    
    dW = np.dot(X.T, norm_output)
    dW = dW/n + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

