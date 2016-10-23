'''
start wiht lddmm landmark matching in tensor flow
'''

import tensorflow as tf

# first get the size of the dataset
N = 2 # number of landmarks
D = 2 # dimension of landmarks
nT = 10 # number of timesteps (integer)
T = 1.0 # length of time
sigmaV = 1.0
sigmaM = 1.0
sigmaR = 1.0
niter = 1000
epsilon = 1.0e-2

# set up initial points
q0 = tf.placeholder(tf.float32,[N,D]) # I will have to feed this in
qtarget = tf.placeholder(tf.float32,[N,D]) # target


q0value = [[1.0, 0.0],[0.0,0.0]]
qtargetvalue = [[2.0,1.0],[-1.0,-1.0]]

# set up variable to optimize, this is momentum
p0 = tf.Variable(tf.zeros([N,D])) # I will optimize over this

sigmaV2 = tf.constant(sigmaV**2)
sigmaM2 = tf.constant(sigmaM**2)
sigmaR2 = tf.constant(sigmaR**2)

# this will be useful for pairwise distances, although I'd bet there is a placeholder

# loop over time
# I may want to save state variables to output them
def EPDiffPointsGausianKernel(q0,p0,sigmaV2,nT=10,T=1.0):
    # for some reason making this into a function seems to have fucked things up
    dt = tf.constant(T/nT)
    qt = [q0]
    pt = [p0]

    qdott = []
    pdott = []

    for t in xrange(nT):
        q = qt[-1]
        p = pt[-1]
        # calculate qdot
        qexpand = tf.expand_dims(q,1) # one column
        qTexpand = tf.expand_dims(q,0) # one row
        qtile = tf.tile(qexpand,[1,N,1])
        qTtile = tf.tile(qTexpand,[N,1,1])
        deltaQ = qtile - qTtile
        deltaQ2 = deltaQ*deltaQ
        d2Q = tf.reduce_sum(deltaQ2,2)
        # gaussian radial kernel
        K = tf.exp(-d2Q/2.0/sigmaV2)

        # now we multiply
        qdot = tf.matmul(K,p)
        qdott.append(qdot)
        
        # and for p
        pdotp = tf.matmul(p,tf.transpose(p)) # nxd times dxn = nxn
        K_pdotp_oosigma2 = (K*pdotp)/sigmaV2
        K_pdotp_oosigma2_expand = tf.expand_dims(K,-1)
        K_pdotp_oosigma2_tile = tf.tile(K_pdotp_oosigma2_expand,[1,1,D])
    
        pdot = tf.reduce_sum(K_pdotp_oosigma2_tile*deltaQ,1)
        pdott.append(pdot)


        # update
        qt.append(q + qdot*dt)
        pt.append(p + pdot*dt)


    return qt,pt,qdott,pdott


# I could put this into a function
qt,pt,qdott,pdott = EPDiffPointsGausianKernel(q0,p0,sigmaV2,nT,T)
q = qt[-1]
p = pt[-1]
qdot0 = qdott[0]

# now we implement a cost function
error = tf.reduce_sum((q-qtarget)**2)/2.0/sigmaM2 + tf.reduce_sum(qdot0*p0)/2.0/sigmaR2

# now we've got to optimize it
train = tf.train.GradientDescentOptimizer(epsilon).minimize(error)


feed_dict={q0:q0value,qtarget:qtargetvalue}
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for _ in range(niter):
        to_evaluate = [train,error]
        out = sess.run(to_evaluate,feed_dict=feed_dict)
        if _%10 == 0:
            print('error: {}'.format(out[1]))
    to_evaluate = [qt,pt]
    out_final = sess.run(to_evaluate,feed_dict=feed_dict)

    

# okay that was super easy!!!!!!

# now that I have a simple function I can easily build more complex trajectories
