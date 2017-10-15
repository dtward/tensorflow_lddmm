import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


import EPDiffPoints as epd
import FlowPoints as fp
import utilities as ut
#import importlib
#importlib.reload(EPDiffPoints)

reload(epd)
reload(fp)
reload(ut)


# first get the size of the dataset
nT = 10 # number of timesteps (integer)
T = 1.0 # length of time
sigmaV = 1.0
sigmaM = 1e-1
sigmaR = 1.0
niter = 1000
epsilon = 1.0e-3



q0value = np.array([[1.0, 0.0],[0.0,0.0],[0.0,1.0]],dtype=np.float32)
qtargetvalue = np.array([[2.0,1.0],[-1.0,-1.0],[0.0,1.0]],dtype=np.float32)
N = q0value.shape[0]
D = q0value.shape[1]

# set up initial points
q0 = tf.placeholder(tf.float32,[N,D]) # I will have to feed this in
qtarget = tf.placeholder(tf.float32,[N,D]) # target


# set up variable to optimize, this is momentum
p0 = tf.Variable(tf.zeros([N,D])) # I will optimize over this


# I could put this into a function
qt,pt,qdott,pdott = epd.EPDiffPointsGausianKernel(q0,p0,sigmaV,nT,T)
q = qt[-1]
p = pt[-1]
qdot0 = qdott[0]

# now we implement a cost function
error = tf.reduce_sum((q-qtarget)**2)/2.0/(sigmaM**2) + tf.reduce_sum(qdot0*p0)/2.0/(sigmaR**2)

# now we've got to optimize it
train = tf.train.GradientDescentOptimizer(epsilon).minimize(error)

feed_dict={q0:q0value,qtarget:qtargetvalue}
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for _ in range(niter):
        to_evaluate = [train,error]
        out = sess.run(to_evaluate,feed_dict=feed_dict)
        if _%10 == 0:
            print('iteration: {}, cost function: {}'.format(_,out[1]))
    to_evaluate = [qt,pt]
    #out_final = sess.run(to_evaluate,feed_dict=feed_dict)
    # this line works on my laptop but not at work
    to_evaluate = qt[1:]
    to_evaluate.extend(pt)
    out_final = sess.run(to_evaluate,feed_dict=feed_dict)
    # and put it back into the form I had before

    qtout = [q0value]
    qtout.extend(out_final[:nT])
    out_final = [qtout, out_final[nT:]]
    

    

    # get a set of points and flow them
    ngrid = 30
    x = tf.linspace(-2.0,2.0,ngrid)
    y = tf.linspace(-2.0,2.0,ngrid)
    xe = tf.expand_dims(x,0)
    xee = tf.expand_dims(xe,2)
    xeetile = tf.tile(xee,[ngrid,1,1])
    ye = tf.expand_dims(y,1)
    yee = tf.expand_dims(ye,2)
    yeetile = tf.tile(yee,[1,ngrid,1])
    xygrid = tf.concat(2,[xeetile,yeetile]) # okay this one does not work with broadcasting
    

    xy = tf.reshape(xygrid,[-1, 2]) # note -1 means calculate the size inorder to fix number of elements
    xyt = fp.FlowOtherPointsGaussianKernel(out_final[0],out_final[1],xy,sigmaV,T)

    xytvalues = sess.run(xyt)
    
# I'd now like to plot xytvalues
# okay that was super easy!!!!!!



plt.close('all')
plt.scatter(xytvalues[-1][:,0],xytvalues[-1][:,1],color='k')


qplot = np.array([tmp for tmp in out_final[0]],dtype=np.float32)

for i in xrange(N):
    plt.scatter(qplot[:,i,0],qplot[:,i,1],color='b')

plt.scatter(qtargetvalue[:,0],qtargetvalue[:,1],color='r')
plt.pause(0.1) # this will draw it

# AWESOME IT WORKS
