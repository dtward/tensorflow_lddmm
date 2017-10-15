# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:21:37 2016

@author: dtward
"""

import tensorflow as tf

def InterpolateImage(Xi,Yi,I,Xo,Yo):
    '''
    Perform linear interpolation on an image
    
    Inputs Xi, Yi are sample points of the current image, they are assumed to 
    be uniformly spaced (i.e. meshgrid)
    
    I is an image to be deformed
    
    Xo, Yo are the points we want to resample the image at
    
    
    '''
    
    Xi = tf.convert_to_tensor(Xi)
    Yi = tf.convert_to_tensor(Yi)
    I = tf.convert_to_tensor(I)
    Xo = tf.convert_to_tensor(Xo)
    Yo = tf.convert_to_tensor(Yo)
    
    # next we have to find xo and dx
    x0 = Xi[0]
    y0 = Yi[0]
    dx = Xi[0,1] - Xi[0,0]
    dy = Yi[1,0] - Yi[0,0]
    
    # next we convert to index
    Xi_ind = (Xi - x0)/dx
    Yi_ind = (Yi - y0)/dx
    Xi_ind_0 = tf.floor(Xi_ind)
    Yi_ind_0 = tf.floor(Yi_ind)
    Xi_ind_p = Xi_ind_0 - Xi_ind_int
    Yi_ind_p = Yi_ind_0 - Yi_ind_int
    
    # boundary conditions
    
    # convert to 1D and then use gather
    # or keep as nd and use gather_nd
    #nY = 
        
    # resample I
    
    
    pass