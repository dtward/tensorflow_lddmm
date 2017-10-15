import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import EPDiffPoints as epd
import FlowPoints as fp
import utilities as ut
#import importlib
#importlib.reload(EPDiffPoints)

reload(epd)
reload(fp)
reload(ut)


# first get the size of the dataset
T = 1.0 # length of time
sigmaV = 1.0
sigmaM = 1e-1
sigmaR = 1.0
niter = 2000
epsilon = 1.0e-5
ntintegrate = 10 # timesteps to integrate
nT = 3 # number of time points
ngrid = 30 # for visualizing

# now that I have a simple function I can easily build more complex trajectories

# TO DO
# timeseries
# then periodic timeseries

# okay timeseries first
# let's begin by making a dataset
# I use the word "value" because q will be a node in the graph
q0value = np.array([[1.0, 0.0],[0.0,0.0],[0.0,1.0]],dtype=np.float32)

N = q0value.shape[0]
D = q0value.shape[1]

# two flow variabiles that will be optimized over
p0 = tf.Variable(tf.zeros([N,D]))
p1 = tf.Variable(tf.zeros([N,D]))


# add time at the beginning, this will work with numpy broadcasting

tvalue = np.array(range(nT),dtype=np.float32)
timeamplitude = 1.0;
randamplitude = 0.1
#qtargetvalue = np.array([[2.0,1.0],[-1.0,-1.0],[0.0,1.0]],dtype=np.float32)[None,:,:] + amplitude*np.sin(tvalue*2.0*np.pi/float(nT))[:,None,None]
#qtargetvalue = np.array([[1.1,0.9],[-0.1,0.1],[0.0,0.1]],dtype=np.float32)[None,:,:] + amplitude*(tvalue - np.mean(tvalue))[:,None,None]
qtargetvalue = np.array(q0value,dtype=np.float32)[None,:,:] + timeamplitude*(tvalue - np.mean(tvalue))[:,None,None] + np.random.randn(nT,N,D)*randamplitude
# these are the nodes, they will be in a feed
q0 = tf.placeholder(tf.float32,[N,D]) # add the one at the end so numpy broadcasting will work
qtarget = [tf.placeholder(tf.float32,[N,D]) for _ in range(nT)]


# now we build a model!
qt,pt,qdott,pdott = epd.EPDiffPointsGausianKernel(q0,p0,sigmaV,ntintegrate,T)
# and now the geodesic part
qqt = []
ppt = []
qqdott = []
ppdott = []
for i in range(nT):
    qt_,pt_,qdott_,pdott_ = epd.EPDiffPointsGausianKernel(qt[-1],p1,sigmaV,ntintegrate,float(tvalue[i]))
    qqt.append(qt_)
    ppt.append(pt_)
    qqdott.append(qdott_)
    ppdott.append(pdott_)
# now we implement a cost function
#error = tf.reduce_sum((qqt-qtarget)**2)/2.0/(sigmaM**2) + tf.reduce_sum(qdot0*p0)/2.0/(sigmaR**2)
error = tf.reduce_sum(qdott[0]*pt[0])/2.0/(sigmaR**2)
for i in range(nT):
    error = error + tf.reduce_sum(qqdott[i][0]*ppt[i][0])/2.0/(sigmaR**2) + tf.reduce_sum((qqt[i][-1] - qtarget[i])**2)/2.0/(sigmaM**2)
    
# optimize
train = tf.train.GradientDescentOptimizer(epsilon).minimize(error)









# now we're going to run it
# first fill in the data
feed_dict={q0:q0value}
for i in range(nT):
    feed_dict[qtarget[i]] = qtargetvalue[i]
    
# start the session
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.initialize_all_variables())

    # start gradient descent steps
    for _ in range(niter):
        to_evaluate = [train,error]
        out = sess.run(to_evaluate,feed_dict=feed_dict)
        if _%10 == 0:
            print('iteration: {}, cost function: {}'.format(_,out[1]))
    # THAT'S IT, EVERYTHING BELOW IS VISUALIZATION            
            
            
    print('p0 is {}'.format(sess.run(p0,feed_dict=feed_dict)))
    print('p1 is {}'.format(sess.run(p1,feed_dict=feed_dict)))
    
    
    for vis_loop in range(nT+1):
        if vis_loop == 0:
            # produce outputs for visualization
            to_evaluate = [qt,pt]
            #out_final = sess.run(to_evaluate,feed_dict=feed_dict)
            # this line works on my laptop but not at work
            to_evaluate = qt[1:]
            to_evaluate.extend(pt)
        else:
            #raise Exception
            to_evaluate = qqt[vis_loop-1]
            to_evaluate.extend(ppt[vis_loop-1])
        out_final = sess.run(to_evaluate,feed_dict=feed_dict)
        # and put it back into the form I had before
        if vis_loop == 0:
            qtout = [q0value]
            qtout.extend(out_final[:ntintegrate])
            out_final = [qtout, out_final[ntintegrate:]]
        else:
            qtout = out_final[:ntintegrate+1]
            out_final = [qtout, out_final[ntintegrate+1:]]
    
        
    
        # get a set of points and flow them
        x = tf.linspace(-2.0,2.0,ngrid)
        y = tf.linspace(-2.0,2.0,ngrid)
        xe = tf.expand_dims(x,0)
        xee = tf.expand_dims(xe,2)
        xeetile = tf.tile(xee,[ngrid,1,1])
        ye = tf.expand_dims(y,1)
        yee = tf.expand_dims(ye,2)
        yeetile = tf.tile(yee,[1,ngrid,1])
        xygrid = tf.concat(2,[xeetile,yeetile]) 
        
    
        xy = tf.reshape(xygrid,[-1, 2]) # note -1 means calculate the size inorder to fix number of elements
        if vis_loop == 0:
            xyt = fp.FlowOtherPointsGaussianKernel(out_final[0],out_final[1],xy,sigmaV,T)
        else:
            xyt = fp.FlowOtherPointsGaussianKernel(out_final[0],out_final[1],xy,sigmaV,float(tvalue[vis_loop-1]))
    
        xytvalues = sess.run(xyt)
        
        # I'd now like to plot xytvalues
        # okay that was super easy!!!!!!
        
        
        
        #plt.close('all')
        plt.figure()
        plt.scatter(xytvalues[-1][:,0],xytvalues[-1][:,1],color='k')
        
        
        qplot = np.array([tmp for tmp in out_final[0]],dtype=np.float32)
        
        for i in xrange(N):
            plt.scatter(qplot[:,i,0],qplot[:,i,1],color='b')
        
        for i in range(nT):
            plt.scatter(qtargetvalue[i,:,0],qtargetvalue[i,:,1],color='r')
        plt.pause(0.1) # this will draw it
        plt.show()