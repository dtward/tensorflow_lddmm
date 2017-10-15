''' 
in periodic version we optimize over sine and cosine weights on the second trajectory

the time needs to be a feed variable here so that I can visualize at any time

I need to think about how to achieve this, I may need to do some serious planning

Maybe I will need two different graphs?  One for generating and one for training?

I shouldn't need that.

I just need to work out the sizes very carefully.

I should have a model output node.  And then I should have a cost node.

The model output should have a size that depends on the input.

The input should be a set of TIMES.

For training, the input is a set of times AND a set of targets.

This means my sine and cosine need to be done in tensorflow, not in numpy

This will be done in version 2




'''

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
epsilonp = 1.0e-5
epsiloncs = 1.0e-4 # I had to shrink this when moving to 5 points
ntintegrate = 10 # timesteps to integrate
nT = 5 # number of time points
ngrid = 30 # for visualizing
M = 2 # number of modes (no 0 frequeny mode, this is a separate trajectory p0)

# now that I have a simple function I can easily build more complex trajectories



# let's begin by making a dataset
# this is an initial guess
# I use the word "value" because q will be a node in the graph
q0value = np.array([[0.5, 0.0],[-0.5,0.0],[0.0,0.6]],dtype=np.float32)

N = q0value.shape[0]
D = q0value.shape[1]

# two flow variabiles that will be optimized over
t0 = tf.Variable(tf.zeros(1))


# add time at the beginning, this will work with numpy broadcasting

# these should be in 0,2*pi
tvalue = np.array(range(nT),dtype=np.float32)/float(nT)*2.0*np.pi

# the coefficients
# there should be M D dimensional coefficients PER VERTEX
# note there is no nT here
#s = tf.Variable(tf.zeros([N,D,M]))
#c = tf.Variable(tf.zeros([N,D,M]))

# a matrix of sines and cosines FOR EACH TIMEPOINT
# should be size (D,M) for each timepoint
# then p1 should be (N,D) for each timepoint
# be contracting over M?




#S = []
#C = []
#p1 = []
#for tv in tvalue:
#    S.append(  np.array([np.array(np.sin(tv*i)) for i in range(1,M+1)]).transpose() )
#    C.append(  np.array([np.array(np.cos(tv*i)) for i in range(1,M+1)]).transpose() )
#    p1.append( tf.matmul(S[-1],s) + tf.matmul(C[-1],c) )
    
    
# okay let's do this again
# p1 is size N, D, nT
# coefficients are size  N, D, M
# sine cosine matrix maps between the two
# note 
# at vertex i, dimension j, timepoint k
# pijk = sum_l sin(2*pi*tk*l)*cijl
c = []
s = []
p1 = []
C = []
S = []
t = []
for i in range(nT):
    # sine matrix
    ti = tvalue[i]
    ci = tf.Variable(tf.zeros([N,D,M]))
    Ci = np.cos( ti * np.array([ m for m in range(1,M+1)]) )
    si = tf.Variable(tf.zeros([N,D,M]))
    Si = np.sin( ti * np.array([ m for m in range(1,M+1)]) )
    # just product and sum over last axis
    p1i = tf.reduce_sum(si*Si + ci*Ci,reduction_indices=[2])
    # append to the list
    c.append(ci)
    s.append(si)
    p1.append(p1i)
    t.append(tf.placeholder(tf.float32))
    # I shouldn't need to save these
    C.append(Ci)
    S.append(Si)
    print(Ci)
    #print(Si)



   

# flow variables are no longer optimized over
p0 = tf.Variable(tf.zeros([N,D]))




timeamplitude = 2.0;
randamplitude = 0.1
#qtargetvalue = np.array([[2.0,1.0],[-1.0,-1.0],[0.0,1.0]],dtype=np.float32)[None,:,:] + amplitude*np.sin(tvalue*2.0*np.pi/float(nT))[:,None,None]
#qtargetvalue = np.array([[1.1,0.9],[-0.1,0.1],[0.0,0.1]],dtype=np.float32)[None,:,:] + amplitude*(tvalue - np.mean(tvalue))[:,None,None]
#qtargetvalue = np.array(q0value,dtype=np.float32)[None,:,:] + timeamplitude*(tvalue**2 - np.mean(tvalue))[:,None,None] + np.random.randn(nT,N,D)*randamplitude
#qtargetvalue = np.array(q0value,dtype=np.float32)[None,:,:] + timeamplitude*(tvalue - np.mean(tvalue))[:,None,None] + np.random.randn(nT,N,D)*randamplitude
# obviously I need a better target for a periodic trajectory
qtargetvalue = np.zeros([nT,N,D],dtype = np.float32)
for i in range(nT):
    qtargetvalue[i,:,:] = q0value + np.array([np.cos(tvalue[i]),np.sin(tvalue[i])])*timeamplitude + np.random.randn(D)*randamplitude



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
    qt_,pt_,qdott_,pdott_ = epd.EPDiffPointsGausianKernel(qt[-1],p1[i],sigmaV,ntintegrate,T)
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
cs = []
cs.extend(c)
cs.extend(s)
train_cs = tf.train.GradientDescentOptimizer(epsiloncs).minimize(error,var_list=cs)
#train_c = tf.train.GradientDescentOptimizer(epsiloncs).minimize(error,var_list=c)
#train_s = tf.train.GradientDescentOptimizer(epsiloncs).minimize(error,var_list=s)
train_p = tf.train.GradientDescentOptimizer(epsilonp).minimize(error,var_list=[p0])











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
        to_evaluate = [train_p,train_cs,error]
        out = sess.run(to_evaluate,feed_dict=feed_dict)
        if _%10 == 0:
            print('iteration: {}, cost function: {}'.format(_,out[-1]))
    # THAT'S IT, EVERYTHING BELOW IS VISUALIZATION            
            
            
    print('p0 is {}'.format(sess.run(p0,feed_dict=feed_dict)))
    print('c is {}'.format(sess.run(c,feed_dict=feed_dict)))
    print('s is {}'.format(sess.run(s,feed_dict=feed_dict)))
    
    
    
    
    for vis_loop in range(nT+1):
        # this section is complicated because the very first time, the first 
        # entry in the list is not a tensor, its a numpy array
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
        xyt = fp.FlowOtherPointsGaussianKernel(out_final[0],out_final[1],xy,sigmaV,T)
        
    
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
        plt.axis('equal')
        plt.axis('square')
        plt.pause(0.1) # this will draw it
        plt.show()