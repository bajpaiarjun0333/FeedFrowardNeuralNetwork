import numpy as np 
import pandas as pd
import math
from sklearn.model_selection import train_test_split

#dataset loading 
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))
x_test=x_test.reshape(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))
x_train=x_train/255
x_test=x_test/255
#create validation set also
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)




class NeuralNetwork:
    def __init__(self):
        self.w=[]
        self.b=[]
        self.a=[]
        self.h=[]
        self.wd=[]
        self.ad=[]
        self.hd=[]
        self.bd=[]
        
    
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
    
    def activations_derivative(self,activation,z):
        if activation=='sigmoid':
            return np.multiply((1/(1+np.exp(-z))),(1-(1/(1+np.exp(-z)))))
        elif activation=='relu':
            return z*(z>0)
        elif activation=='tanh':
            y=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
            return 1-np.square(y)
        
    def loss_function(self,loss_fn,yhat,y_train):
        if loss_fn=='cross_entropy':
            sum=0
            for i in range(y_train.shape[0]):
                sum+=-((np.log2(yhat[i][y_train[i]])))
            return sum
        if loss_fn=='mean_square':
            return (yhat-y_train)**2


    def make_layers(self,no_of_hidden_layers,no_of_neuron,input_neuron,initialization,no_of_classes):
        layer=[]
        layer.append(input_neuron)
        for i in range(no_of_hidden_layers):
            layer.append(no_of_neuron)
        layer.append(no_of_classes)
        if initialization=='random':
            for i in range(no_of_hidden_layers+1):
                weights=np.random.uniform(-1,1,(layer[i],layer[i+1]))
                bias=np.random.uniform(-1,1,(1,layer[i+1]))
                self.w.append(weights)
                self.b.append(bias)
        if initialization=='xavier':
            for i in range(no_of_hidden_layers+1):
                n=np.sqrt(6/(layer[i]+layer[i+1]))
                weights=np.random.uniform(-n,n,(layer[i],layer[i+1]))
                bias=np.random.uniform(-n,n,(1,layer[i+1]))
                self.w.append(weights)
                self.b.append(bias)

    def forward_pass(self,x,activation):
        self.a=[]
        self.h=[]
        temp=x
        l=len(self.w)
        for i in range (l-1):
            a1=np.add(np.matmul(temp,self.w[i]),self.b[i])
            h1=self.activations(activation,a1)
            self.a.append(a1)
            self.h.append(h1)
            temp=h1
        a1=np.add(np.matmul(temp,self.w[l-1]),self.b[l-1])
        h1=self.activations('softmax',a1)
        self.a.append(a1)
        self.h.append(h1)
    
    def backward_pass(self,yhat,y_train,x_train,no_of_classes,activation):
        self.wd=[]
        self.bd=[]
        self.ad=[]
        self.hd=[]
        el=np.zeros((y_train.shape[0],no_of_classes))
        for i in range (y_train.shape[0]):
            el[i][y_train[i]]=1

        yhatl=np.zeros((yhat.shape[0],1))
        for i in range (yhat.shape[0]):
            yhatl[i]=yhat[i][y_train[i]]

        hd1=-(el/yhatl)
        ad1=-(el-yhat)
        self.hd.append(hd1)
        self.ad.append(ad1)
        l=len(self.w)
        for i in range(l-1,-1,-1):
            q=self.h[i-1].T
            if i==0:
                q=x_train.T
            wi=np.matmul(q,self.ad[len(self.ad)-1])/x_train.shape[0]
            bi=np.sum(self.ad[len(self.ad)-1],axis=0)/x_train.shape[0]
            if i!=0:
                hd1=np.matmul(self.ad[len(self.ad)-1],self.w[i].T)
                der=self.activations_derivative(activation,self.a[i-1])
                ad1=np.multiply(hd1,der)
                self.hd.append(hd1)
                self.ad.append(ad1)
            self.wd.append(wi)
            self.bd.append(bi)
        
    
    def accuracy(self,x_test,y_test,activation):
        self.forward_pass(x_test,activation)
        l=len(self.w)
        ypred=np.argmax(self.h[l-1],axis=1)
        count=0
        for i in range(y_test.shape[0]):
            if ypred[i]!=y_test[i]:
                count=count+1
        return ((x_test.shape[0]-count)/y_test.shape[0])
    
    def createBatches(self,x_train,y_train,batchSize):
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
    
    def onePass(self,x_train,y_train,no_of_classes,l,n,activation):
        self.forward_pass(x_train,activation)
        self.backward_pass(self.h[l-1],y_train,x_train,no_of_classes,activation)
    
    def batch(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,activation,loss_fn):
        data,ans=self.createBatches(x_train,y_train,batchSize)

        for i in range(iter):
            h=None
            for j in range(len(data)):
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation)
                for j in range (l):
                    self.w[j]=self.w[j]-n*self.wd[l-1-j]
                    self.b[j]=self.b[j]-n*self.bd[l-1-j]

            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train)/x_train.shape[0]
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,h[l-1],y_val)/x_val.shape[0]
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            print("Iteration Number: "+str(i)+" Train loss: "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validaion loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            

    def momentum(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn):
        data,ans=self.createBatches(x_train,y_train,batchSize)
  
        moment=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((self.w[i].shape))
            temp2=np.zeros(self.b[i].shape)
            moment.append(temp)
            momentB.append(temp2)

        for i in range(iter):
            for j in range(len(data)):
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation)
                for k in range (l):
                    moment[k]=(moment[k]*beta)+self.wd[l-1-k]
                    momentB[k]=(momentB[k]*beta)+self.bd[l-1-k]
                    self.w[k]=self.w[k]-n*moment[k]
                    self.b[k]=self.b[k]-n*momentB[k]
  
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train)/x_train.shape[0]
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val)/x_val.shape[0]
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            print("Iteration Number: "+str(i)+" Train loss: "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validaion loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def nestrov(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn):
        data,ans=self.createBatches(x_train,y_train,batchSize)

        moment=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((self.w[i].shape))
            temp2=np.zeros((self.b[i].shape))
            moment.append(temp)
            momentB.append(temp2)
    
        for i in range(iter):
            for j in range(len(data)):
                for k in range(l):
                    self.w[k]=self.w[k]-beta*moment[k]
                    self.b[k]=self.b[k]-beta*momentB[k]
                self.onePass(data[k],ans[k],no_of_classes,l,n,activation)
                for k in range (l):
                    moment[k]=beta*moment[k]+n*self.wd[l-1-k]
                    momentB[k]=beta*momentB[k]+n*self.bd[l-1-k]
                    self.w[k]=self.w[k]-moment[k]
                    self.b[k]=self.b[k]-momentB[k]
      
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train)/x_train.shape[0]
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val)/x_val.shape[0]
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            print("Iteration Number: "+str(i)+" Train loss: "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validaion loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def rmsProp(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn):
        data,ans=self.createBatches(x_train,y_train,batchSize)

        momentW=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((self.w[i].shape))
            temp2=np.zeros((self.b[i].shape))
            momentW.append(temp)
            momentB.append(temp2)

        epsilon=1e-8
        for i in range(int(iter)):
            for j in range(len(data)):
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation)
                for k in range (l):
                    momentW[k]=(momentW[k]*beta)+(1-beta)*np.square(self.wd[l-1-k])
                    momentB[k]=(momentB[k]*beta)+(1-beta)*np.square(self.bd[l-1-k])
                    self.w[k]=self.w[k]-(n/np.sqrt(np.linalg.norm(momentW[k]+epsilon)))*self.wd[l-1-k]
                    self.b[k]=self.b[k]-(n/np.sqrt(np.linalg.norm(momentB[k]+epsilon)))*self.bd[l-1-k]
  
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train)/x_train.shape[0]
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val)/x_val.shape[0]
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            print("Iteration Number: "+str(i)+" Train loss: "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validaion loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def adam(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta1,beta2,activation,loss_fn):
        data,ans=self.createBatches(x_train,y_train,batchSize)

        mt_w=[]
        vt_w=[]
        mt_b=[]
        vt_b=[]
        for i in range(l):
            temp=np.zeros((self.w[i].shape))
            temp2=np.zeros((self.w[i].shape))
            mt_w.append(temp)
            vt_w.append(temp2)
            temp=np.zeros((self.b[i].shape))
            temp2=np.zeros((self.b[i].shape))
            mt_b.append(temp)
            vt_b.append(temp2)

        epsilon=1e-10
        t=0
        for i in range(int(iter)):
            for j in range(len(data)):
                t=t+1
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation)
                for k in range (l):
                    mt_w[k]=(mt_w[k]*beta1)+(1-beta1)*self.wd[l-1-k]
                    mt_w_hat=mt_w[k]/(1-beta1**t)
                    vt_w[k]=(vt_w[k]*beta2)+(1-beta2)*np.square(self.wd[l-1-k])
                    vt_w_hat=vt_w[k]/(1-beta2**t)
                    mt_b[k]=(mt_b[k]*beta1)+(1-beta1)*self.bd[l-1-k]
                    mt_b_hat=mt_b[k]/(1-beta1**t)
                    vt_b[k]=(vt_b[k]*beta2)+(1-beta2)*np.square(self.bd[l-1-k])
                    vt_b_hat=vt_b[k]/(1-beta2**t)
                    self.w[k]=self.w[k]-(n/np.sqrt(np.linalg.norm(vt_w_hat+epsilon)))*mt_w_hat
                    self.b[k]=self.b[k]-(n/np.sqrt(np.linalg.norm(vt_b_hat+epsilon)))*mt_b_hat
            
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train)/x_train.shape[0]
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val)/x_val.shape[0]
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            print("Iteration Number: "+str(i)+" Train loss: "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validaion loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def Nadam(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta1,beta2,activation,loss_fn):
        data,ans=self.createBatches(x_train,y_train,batchSize)
        mt_w=[]
        vt_w=[]
        mt_b=[]
        vt_b=[]
        for i in range(l):
            temp=np.zeros((self.w[i].shape))
            temp2=np.zeros((self.w[i].shape))
            mt_w.append(temp)
            vt_w.append(temp2)
            temp=np.zeros((self.b[i].shape))
            temp2=np.zeros((self.b[i].shape))
            mt_b.append(temp)
            vt_b.append(temp2)

        epsilon=1e-10
        t=0
        for i in range(int(iter)):
            for j in range(len(data)):
                t=t+1
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation)
                for k in range (l):
                    mt_w[k]=(mt_w[k]*beta1)+(1-beta1)*self.wd[l-1-k]
                    mt_w_hat=mt_w[k]/(1-beta1**t)
                    vt_w[k]=(vt_w[k]*beta2)+(1-beta2)*np.square(self.wd[l-1-k])
                    vt_w_hat=vt_w[k]/(1-beta2**t)
                    mt_b[k]=(mt_b[k]*beta1)+(1-beta1)*self.bd[l-1-k]
                    mt_b_hat=mt_b[k]/(1-beta1**t)
                    vt_b[k]=(vt_b[k]*beta2)+(1-beta2)*np.square(self.bd[l-1-k])
                    vt_b_hat=vt_b[k]/(1-beta2**t)
                    self.w[k]=self.w[k]-(n/np.sqrt(np.linalg.norm(vt_w_hat+epsilon)))*(beta1*mt_w_hat+(((1-beta1)*self.wd[l-1-k])/(1-beta1**t)))
                    self.b[k]=self.b[k]-(n/np.sqrt(np.linalg.norm(vt_b_hat+epsilon)))*(beta1*mt_b_hat+(((1-beta1)*self.bd[l-1-k])/(1-beta1**t)))
  
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train)/x_train.shape[0]
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val)/x_val.shape[0]
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            print("Iteration Number: "+str(i)+" Train loss: "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validaion loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def architecture(self,x_train,y_train,x_val,y_val,no_of_classes,no_of_hidden_layers,no_of_neuron,input_neuron,batchSize,initialization,loss_fn,activation,optimizer,n,iter):
        self.w=[]
        self.b=[]
        self.make_layers(no_of_hidden_layers,no_of_neuron,input_neuron,initialization,no_of_classes)
        l=len(self.w)
        if optimizer=="batch":
            self.batch(x_train,y_train,no_of_classes,l,iter,n,batchSize,activation,loss_fn)
        if optimizer=='momentum':
            self.momentum(x_train,y_train,no_of_classes,l,iter,n,batchSize,0.9,activation,loss_fn)
        if optimizer=='nestrov':
            self.nestrov(x_train,y_train,no_of_classes,l,iter,n,batchSize,0.9,activation,loss_fn)
        if optimizer=='rmsProp':
            self.rmsProp(x_train,y_train,no_of_classes,l,iter,n,batchSize,0.9,activation,loss_fn)
        if optimizer=='adam':
            self.adam(x_train,y_train,no_of_classes,l,iter,n,batchSize,0.9,0.999,activation,loss_fn)
        if optimizer=='Nadam':
            self.Nadam(x_train,y_train,no_of_classes,l,iter,n,batchSize,0.9,0.999,activation,loss_fn)



obj=NeuralNetwork()
obj.architecture(x_train,y_train,x_test,y_test,10,3,128,784,32,'xavier','cross_entropy','tanh','Nadam',0.01,5)
