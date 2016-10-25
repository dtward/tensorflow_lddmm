'''
start wiht lddmm landmark matching in tensor flow
'''

import tensorflow as tf
import utilities as ut

# I may want to save state variables to output them
# I can always take N and D as an input, but really I want to get them from q0!


def EPDiffPointsGausianKernel(q0,p0,sigmaV,nT=10,T=1.0):
    dt = tf.constant(T/nT)
    qt = [q0]
    pt = [p0]

    qdott = []
    pdott = []

    for t in xrange(nT):
        q = qt[-1]
        p = pt[-1]
        # note that tensorflow uses numpy style broadcasting
        # calculate qdot
        deltaQ = ut.pairwiseDifference(q,q)
        deltaQ2 = deltaQ*deltaQ
        d2Q = tf.reduce_sum(deltaQ2,2)
        # gaussian radial kernel
        K = tf.exp(-d2Q/2.0/(sigmaV**2))

        # now we multiply
        qdot = tf.matmul(K,p)
        qdott.append(qdot)
        
        # and for p
        pdotp = tf.matmul(p,tf.transpose(p)) # nxd times dxn = nxn
        K_pdotp_oosigma2 = (K*pdotp)/(sigmaV**2)
        K_pdotp_oosigma2_expand = tf.expand_dims(K,-1)
        K_pdotp_oosigma2_expand_deltaQ = K_pdotp_oosigma2_expand*deltaQ
        # for some reason this isn't workin gnow
        # deltaQ size is 3x3x2
        # K size is 3x3x1
        pdot = tf.reduce_sum(K_pdotp_oosigma2_expand_deltaQ,1)
        pdott.append(pdot)


        # update
        qt.append(q + qdot*dt)
        pt.append(p + pdot*dt)


    return qt,pt,qdott,pdott


