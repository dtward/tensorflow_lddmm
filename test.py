import tensorflow as tf
import EPDiffPoints as epd
import FlowPoints as fp
#import importlib
#importlib.reload(EPDiffPoints)
reload(epd)
reload(fp)

# first get the size of the dataset
N = 2 # number of landmarks
D = 2 # dimension of landmarks
nT = 10 # number of timesteps (integer)
T = 1.0 # length of time
sigmaV = 1.0
sigmaM = 1.0
sigmaR = 2.0
niter = 1000
niter = 100
epsilon = 1.0e-2

# set up initial points
q0 = tf.placeholder(tf.float32,[N,D]) # I will have to feed this in
qtarget = tf.placeholder(tf.float32,[N,D]) # target


q0value = [[1.0, 0.0],[0.0,0.0]]
qtargetvalue = [[2.0,1.0],[-1.0,-1.0]]

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
            print('cost function: {}'.format(out[1]))
    to_evaluate = [qt,pt]
    out_final = sess.run(to_evaluate,feed_dict=feed_dict)

    # get a set of points and flow them
    x = tf.linspace(-2.0,2.0,20)
    y = tf.linspace(-2.0,2.0,20)
    xe = tf.expand_dims(x,0)
    xee = tf.expand_dims(xe,2)
    xeetile = tf.tile(xee,[20,1,1])
    ye = tf.expand_dims(y,1)
    yee = tf.expand_dims(ye,2)
    yeetile = tf.tile(yee,[1,20,1])
    xygrid = tf.concat(2,[xeetile,yeetile]) # okay this one does not work with broadcasting
    

    xy = tf.reshape(xygrid,[-1, 2]) # note -1 means calculate the size inorder to fix number of elements
    xyt = fp.FlowOtherPointsGaussianKernel(out_final[0],out_final[1],xy,sigmaV,T)

    xytvalues = sess.run(xyt)
    
# I'd now like to plot xytvalues
# okay that was super easy!!!!!!
import matplotlib.pyplot as plt

plt.scatter(xytvalues[-1][:,0],xytvalues[-1][:,1])


plt.pause(0.1)

# AWESOME IT WORKS

# now that I have a simple function I can easily build more complex trajectories




