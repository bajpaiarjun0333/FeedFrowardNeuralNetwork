import numpy as np 
import pandas as pd
import math

#dataset loading 
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))
x_test=x_test.reshape(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))
x_train=x_train/255
x_test=x_test/255



class NeuralNetwork:
    def __init__(self):
        self.w=[]
        self.b=[]
        self.a=[]
        self.h=[]
        self.wd=[]
        self.ad=[]
    
    def activations(self,activation,z):
        if activation=='sigmoid':
            return 1/(1+np.exp(-z))
        elif activation=='relu':
            return z*(z>0)
        elif activation=='tanh':
            return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        elif activation=='softmax':
            for i in range(z.shape[0]):
                sum=0
                for j in range(z.shape[1]):
                    sum=sum+np.exp(z[i][j])
            z[i]=np.exp(z[i])/sum
        return z
    
    def activations_derivative(activation,z):
        if activation=='sigmoid':
            return np.multiply((1/(1+np.exp(-z))),(1-(1/(1+np.exp(-z)))))
        elif activation=='relu':
            return z*(z>0)
        elif activation=='tanh':
            y=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
            return 1-np.square(y)
        
    def loss_function(loss_fn,yhat,y_train):
        if loss_fn=='cross_entropy':
            sum=0
            for i in range(y_train.shape[0]):
                sum+=-((np.log2(yhat[i][y_train[i]])))
            return sum
        if loss_fn=='mean_square':
            return (yhat-y_train)**2


    def make_layer(self,w,b,input,output,initialization):
        if initialization=='random':
            weights=np.random.uniform(-1,1,(input,output))
            bias=np.random.uniform(-1,1,(1,output))
            self.w.append(weights)
            self.b.append(bias)
        if initialization=='xavier':
            n=np.sqrt(6/(input+output))
            wweights=np.random.uniform(-n,n,(input,output))
            bias=np.random.uniform(-n,n,(1,output))
            self.w.append(weights)
            self.b.append(bias)

    def forward_pass(x,w,b,activation):
        a=[]
        h=[]
        temp=x
        l=len(w)
        for i in range (l-1):
            a1=np.add(np.matmul(temp,w[i]),b[i])
            h1=self.activations(activation,a1)
            a.append(a1)
            h.append(h1)
            temp=h1
        a1=np.add(np.matmul(temp,w[l-1]),b[l-1])
        h1=self.activations('softmax',a1)
        a.append(a1)
        h.append(h1)
        return (a,h)
    
    def backward_pass(w,b,a,h,yhat,y_train,x_train,no_of_classes,activation):
        wd=[]
        bd=[]
        ad=[]
        hd=[]
        el=np.zeros((y_train.shape[0],no_of_classes))
        for i in range (y_train.shape[0]):
            el[i][y_train[i]]=1

        yhatl=np.zeros((yhat.shape[0],1))
        for i in range (yhat.shape[0]):
            yhatl[i]=yhat[i][y_train[i]]

        hd1=-(el/yhatl)
        ad1=-(el-yhat)
        hd.append(hd1)
        ad.append(ad1)
        l=len(w)
        for i in range(l-1,-1,-1):
            q=h[i-1].T
            if i==0:
                q=x_train.T
            wi=np.matmul(q,ad[len(ad)-1])/x_train.shape[0]
            bi=np.sum(ad[len(ad)-1],axis=0)/x_train.shape[0]
            if i!=0:
                hd1=np.matmul(ad[len(ad)-1],w[i].T)
                der=activations_derivative(activation,a[i-1])
                ad1=np.multiply(hd1,der)
                hd.append(hd1)
                ad.append(ad1)
            wd.append(wi)
            bd.append(bi)
        return (wd,bd)
    
    def accuracy(x_test,y_test,w,b,activation):
        a,h=forward_pass(x_test,w,b,activation)
        l=len(w)
        ypred=np.argmax(h[l-1],axis=1)
        count=0
        for i in range(y_test.shape[0]):
            if ypred[i]!=y_test[i]:
                count=count+1
        return ((x_test.shape[0]-count)/y_test.shape[0])
    
    def createBatches(x_train,y_train,batchSize):
        data=[]
        ans=[]
        for i in range(math.ceil(x_train.shape[0]/batchSize)):
            batch=[]
            batch_ans=[]
            for j in range(i*batchSize,min((i+1)*batchSize,x_train.shape[0]),1):
                batch.append(x_train[j])
                batch_ans.append(y_train[j])
            batch=np.array(batch)
            batch_ans=np.array(batch_ans)
            data.append(batch)
            ans.append(batch_ans)
        return data,ans
    
    def onePass(x_train,y_train,no_of_classes,w,b,l,n,activation):
        a,h=forward_pass(x_train,w,b,activation)
        wd,bd=backward_pass(w,b,a,h,h[l-1],y_train,x_train,no_of_classes,activation)
        return (wd,bd)
    
    def batchGrad(x_train,y_train,no_of_classes,w,b,l,iter,n,batchSize,activation,loss_fn):
        data,ans=createBatches(x_train,y_train,batchSize)

        for i in range(iter):
            h=None
            for j in range(len(data)):
                wd,bd=onePass(data[j],ans[j],no_of_classes,w,b,l,n,activation)
                for j in range (l):
                    w[j]=w[j]-n*wd[l-1-j]
                    b[j]=b[j]-n*bd[l-1-j]
        a=[]
        h=[]
        a,h=forward_pass(x_train,w,b,activation)
        loss=loss_function(loss_fn,h[l-1],y_train)
        print("Iteration Number: "+str(i)+" loss: "+str(loss/x_train.shape[0]))

    def momentumGradientDescent(x_train,y_train,no_of_classes,w,b,l,iter,n,batchSize,beta,activation,loss_fn):
        data,ans=createBatches(x_train,y_train,batchSize)
  
        moment=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((w[i].shape))
            temp2=np.zeros(b[i].shape)
            moment.append(temp)
            momentB.append(temp2)

        for i in range(iter):
            for j in range(len(data)):
                wd,bd=onePass(data[j],ans[j],no_of_classes,w,b,l,n,activation)
                for k in range (l):
                    moment[k]=(moment[k]*beta)+wd[l-1-k]
                    momentB[k]=(momentB[k]*beta)+bd[l-1-k]
                    w[k]=w[k]-n*moment[k]
                    b[k]=b[k]-n*momentB[k]
  
        a,h=forward_pass(x_train,w,b,activation)
        loss=loss_function(loss_fn,h[l-1],y_train)
        print("Iteration Number: "+str(i)+" loss: "+str(loss/x_train.shape[0]))

    def nsGradientDescent(x_train,y_train,no_of_classes,w,b,l,iter,n,batchSize,beta,activation,loss_fn):
        data,ans=createBatches(x_train,y_train,batchSize)

        moment=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((w[i].shape))
            temp2=np.zeros((b[i].shape))
            moment.append(temp)
            momentB.append(temp2)
    
        for i in range(iter):
            for j in range(len(data)):
                for k in range(l):
                    w[k]=w[k]-beta*moment[k]
                    b[k]=b[k]-beta*momentB[k]
                wd,bd=onePass(data[k],ans[k],no_of_classes,w,b,l,n,activation)
                for k in range (l):
                    moment[k]=beta*moment[k]+n*wd[l-1-k]
                    momentB[k]=beta*momentB[k]+n*bd[l-1-k]
                    w[k]=w[k]-moment[k]
                    b[k]=b[k]-momentB[k]
      
        a,h=forward_pass(x_train,w,b,activation)
        loss=loss_function(loss_fn,h[l-1],y_train)
        print("Iteration Number: "+str(i)+" loss: "+str(loss/x_train.shape[0]))

    def rmsProp(x_train,y_train,no_of_classes,w,b,l,iter,n,batchSize,beta,activation,loss_fn):
        data,ans=createBatches(x_train,y_train,batchSize)

        momentW=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((w[i].shape))
            temp2=np.zeros((b[i].shape))
            momentW.append(temp)
            momentB.append(temp2)

        epsilon=1e-8
        for i in range(int(iter)):
            for j in range(len(data)):
                wd,bd=onePass(data[j],ans[j],no_of_classes,w,b,l,n,activation)
                for k in range (l):
                    momentW[k]=(momentW[k]*beta)+(1-beta)*np.square(wd[l-1-k])
                    momentB[k]=(momentB[k]*beta)+(1-beta)*np.square(bd[l-1-k])
                    w[k]=w[k]-(n/np.sqrt(np.linalg.norm(momentW[k]+epsilon)))*wd[l-1-k]
                    b[k]=b[k]-(n/np.sqrt(np.linalg.norm(momentB[k]+epsilon)))*bd[l-1-k]
  
        a,h=forward_pass(x_train,w,b,activation)
        loss=loss_function(loss_fn,h[l-1],y_train)
        print("Iteration Number: "+str(i)+" loss: "+str(loss/x_train.shape[0]))

def adam(x_train,y_train,no_of_classes,w,b,l,iter,n,batchSize,beta1,beta2,activation,loss_fn):
  data,ans=createBatches(x_train,y_train,batchSize)

  mt_w=[]
  vt_w=[]
  mt_b=[]
  vt_b=[]
  for i in range(l):
    temp=np.zeros((w[i].shape))
    temp2=np.zeros((w[i].shape))
    mt_w.append(temp)
    vt_w.append(temp2)
    temp=np.zeros((b[i].shape))
    temp2=np.zeros((b[i].shape))
    mt_b.append(temp)
    vt_b.append(temp2)

  epsilon=1e-10
  t=0
  for i in range(int(iter)):
    for j in range(len(data)):
      t=t+1
      wd,bd=onePass(data[j],ans[j],no_of_classes,w,b,l,n,activation)
      for k in range (l):
        mt_w[k]=(mt_w[k]*beta1)+(1-beta1)*wd[l-1-k]
        mt_w_hat=mt_w[k]/(1-beta1**t)
        vt_w[k]=(vt_w[k]*beta2)+(1-beta2)*np.square(wd[l-1-k])
        vt_w_hat=vt_w[k]/(1-beta2**t)
        mt_b[k]=(mt_b[k]*beta1)+(1-beta1)*bd[l-1-k]
        mt_b_hat=mt_b[k]/(1-beta1**t)
        vt_b[k]=(vt_b[k]*beta2)+(1-beta2)*np.square(bd[l-1-k])
        vt_b_hat=vt_b[k]/(1-beta2**t)
        w[k]=w[k]-(n/np.sqrt(np.linalg.norm(vt_w_hat+epsilon)))*mt_w_hat
        b[k]=b[k]-(n/np.sqrt(np.linalg.norm(vt_b_hat+epsilon)))*mt_b_hat
  
    a,h=forward_pass(x_train,w,b,activation)
    loss=loss_function(loss_fn,h[l-1],y_train)
    print("Iteration Number: "+str(i)+" loss: "+str(loss/x_train.shape[0]))

    def Nadam(x_train,y_train,no_of_classes,w,b,l,iter,n,batchSize,beta1,beta2,activation,loss_fn):
        data,ans=createBatches(x_train,y_train,batchSize)
        mt_w=[]
        vt_w=[]
        mt_b=[]
        vt_b=[]
        for i in range(l):
            temp=np.zeros((w[i].shape))
            temp2=np.zeros((w[i].shape))
            mt_w.append(temp)
            vt_w.append(temp2)
            temp=np.zeros((b[i].shape))
            temp2=np.zeros((b[i].shape))
            mt_b.append(temp)
            vt_b.append(temp2)

        epsilon=1e-10
        t=0
        for i in range(int(iter)):
            for j in range(len(data)):
                t=t+1
                wd,bd=onePass(data[j],ans[j],no_of_classes,w,b,l,n,activation)
                for k in range (l):
                    mt_w[k]=(mt_w[k]*beta1)+(1-beta1)*wd[l-1-k]
                    mt_w_hat=mt_w[k]/(1-beta1**t)
                    vt_w[k]=(vt_w[k]*beta2)+(1-beta2)*np.square(wd[l-1-k])
                    vt_w_hat=vt_w[k]/(1-beta2**t)
                    mt_b[k]=(mt_b[k]*beta1)+(1-beta1)*bd[l-1-k]
                    mt_b_hat=mt_b[k]/(1-beta1**t)
                    vt_b[k]=(vt_b[k]*beta2)+(1-beta2)*np.square(bd[l-1-k])
                    vt_b_hat=vt_b[k]/(1-beta2**t)
                    w[k]=w[k]-(n/np.sqrt(np.linalg.norm(vt_w_hat+epsilon)))*(beta1*mt_w_hat+(((1-beta1)*wd[l-1-k])/(1-beta1**t)))
                    b[k]=b[k]-(n/np.sqrt(np.linalg.norm(vt_b_hat+epsilon)))*(beta1*mt_b_hat+(((1-beta1)*bd[l-1-k])/(1-beta1**t)))
  
        a,h=forward_pass(x_train,w,b,activation)
        loss=loss_function(loss_fn,h[l-1],y_train)
        print("Iteration Number: "+str(i)+" loss: "+str(loss/x_train.shape[0]))

    












