# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:52:34 2021

@author: JLO033
"""
import numpy as np


"""This creates some vectors"""

"""
x1 = np.array([-1,-1])
x2 = np.array([1,0])
#x3 = np.array([-1,1.5])
x3 = np.array([-1,10])

xx = np.vstack((x1,x2,x3,x1,x2,x3,x1,x2,x3,x1,x2,x3, x1, x2, x3, x1,x2,x3))

y1 = np.array([1])
y2 = np.array([-1])
y3 = np.array([1])

yy = np.vstack((y1,y2,y3,y1,y2,y3,y1,y2,y3,y1,y2,y3,y1,y2,y3,y1,y2,y3))


theta = np.array([0,0])
errs = 0

"""
    

#Perceptron without offset    
    
"""
for i in range(1,17):
    #percp(xx[i],yy[i],theta)
    act = yy[i]*np.dot(theta,xx[i])
    #print("Iteration: ", i) #if starting with x1
    print("Iteration: ", i+1) #if starting with x2
    print("Activation= ", act)
    
    
    if act>0:
        print("CORRECT")
    else:
        theta= theta + yy[i]*xx[i]
        print("WRONG")
        print("New Theta = ", theta)
        errs = errs+1
        print("Total errors: ",errs)
"""    
    
    
def percp(x,y,theta=[0,0,0]):

    errs = 0
    for i in range(len(x)):
        #percp(xx[i],yy[i],theta)
        act = y[i]*(np.dot(theta,x[i]))
        print("Iteration: ", i+1) #if starting with x1
        #print("Iteration: ", i+1) #if starting with x2
        print("Activation= ", act)
        
        
        if act>0:
            print("CORRECT", '\n')
        else:
            theta = theta + y[i]*x[i]
            print("WRONG")
            print("New Theta = ", theta)
            errs = errs+1
            print("Total errors: ",errs, '\n')
    
    
    
    
# Perceptron with offset


"""

for i in range(1,17):
    #percp(xx[i],yy[i],theta)
    act = yy[i]*(np.dot(theta,xx[i])+thetazero)
    #print("Iteration: ", i) #if starting with x1
    print("Iteration: ", i+1) #if starting with x2
    print("Activation= ", act)
    
    
    if act>0:
        print("CORRECT")
    else:
        theta = theta + yy[i]*xx[i]
        thetazero = thetazero + yy[i]
        print("WRONG")
        print("New Theta = ", theta)
        errs = errs+1
        print("Total errors: ",errs)
    
"""    
    
def percpoff(x,y,theta=[0,0],thetazero=0):

    errs = 0
    for i in range(len(x)):
        #percp(xx[i],yy[i],theta)
        act = y[i]*(np.dot(theta,x[i])+thetazero)
        print("Iteration: ", i+1) #if starting with x1
        #print("Iteration: ", i+1) #if starting with x2
        print("Activation= ", act)
        
        
        if act>0:
            print("CORRECT", '\n')
        else:
            theta = theta + y[i]*x[i]
            thetazero = thetazero + y[i]
            print("WRONG")
            print("New Theta = ", theta)
            print("New Theta_Zero = ", thetazero)
            errs = errs+1
            print("Total errors: ",errs, '\n')



def hinge_loss_single(feature_vector, label, theta, theta_0):
    
    #errs = 0
    loss = np.zeros((len(feature_vector),1))
    agreement = np.zeros((len(feature_vector),1))
    for i in range(len(feature_vector)):
        
        agreement[i] = label[i]*(np.dot(theta,feature_vector[i])+theta_0)
        loss[i] = max(0, 1-agreement)
        """
        if agreement<0:
            loss[i] = 1-agreement
            errs = errs -agreement +1
        else:
            loss[i] = 0
        """        
    return (sum(loss)/len(feature_vector))
                
                
            
def hinge_loss_full_old(feature_matrix, labels, theta, theta_0):
        
    errs = 0
    loss = np.zeros((len(feature_matrix),1))
    for i in range(len(feature_matrix)):
        
        agreement = labels[i]*(np.dot(theta,feature_matrix[i])+theta_0)
        if agreement<0:
            loss[i] = 1-agreement
            errs = errs -agreement +1
        else:
            loss[i] = 0
                
        return (errs/len(feature_matrix))
    
    
    
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
        
    loss = np.zeros((len(feature_matrix),1))
    agreement = np.zeros((len(feature_matrix),1))
    for i in range(len(feature_matrix)):
        
        agreement[i] = labels[i]*(np.dot(theta,feature_matrix[i])+theta_0)
        loss[i] = max(0, 1-agreement[i])
        """
        if agreement<0:
            loss[i] = 1-agreement
            errs = errs -agreement +1
        else:
            loss[i] = 0
        """        
    return (sum(loss)/len(feature_matrix))
    




