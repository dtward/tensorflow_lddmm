'''
Flow points given momentum over time
'''

import tensorflow as tf
import utilities as ut

def FlowPointsGaussianKernel(q0,pt,sigmaV,T=1.0):
    '''
    Do I need nT and T here?
    From the size of pt I can get nT
    but I definitely do need T
    '''
    nT = len(pt)
    dt = tf.constant(T/nT)
    qt = [q0]
    for t in xrange(nT):
        q = qt[-1]
        p = pt[t]
        # calculate qdot
        d2 = ut.pairwiseDistanceSquared(q,q)
        # gaussian radial kernel
        K = tf.exp(-d2/2.0/(sigmaV**2))

        # now we multiply
        qdot = tf.matmul(K,p)
        
        # update
        qt.append(q + qdot*dt)
    return qt




def FlowOtherPointsGaussianKernel(qt,pt,x0,sigmaV,T=1.0):
    '''
    move other points after calculatint EPDiff flow
    this will be useful for a control points approach, or for visualizing deformed grids, etc.
    '''
    nT = len(pt)
    dt = tf.constant(T/nT)
    xt = [x0]
    for t in xrange(nT):
        q = qt[t]
        p = pt[t]
        x = xt[-1]
        # calculate qdot
        d2 = ut.pairwiseDistanceSquared(x,q)
        # gaussian radial kernel
        K = tf.exp(-d2/2.0/(sigmaV**2))

        # now we multiply
        xdot = tf.matmul(K,p)
        
        # update
        xt.append(x + xdot*dt)
    return xt

