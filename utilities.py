import tensorflow as tf

def pairwiseDifference(x1,x2):
    ''' x1 and x2 are N1xD and N2xD column vectors '''
    x1expand = tf.expand_dims(x1,1) # expand to n1x1xd
    x2expand = tf.expand_dims(x2,0) # expand to 1xn2xd
    delta = x1expand-x2expand
    return delta

def pairwiseDistanceSquared(x1,x2):
    delta = pairwiseDifference(x1,x2)
    delta2 = delta*delta
    d2 = tf.reduce_sum(delta2,2)
    return d2
    
    
