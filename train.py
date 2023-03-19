import numpy as np 
import pandas as pd
import math
import wandb
import argparse
import os
from sklearn.model_selection import train_test_split

##########################################################   ARG PARSE #########################################################





os.environ['WAND_NOTEBOOK_NAME']='train.py'

parser=argparse.ArgumentParser()

parser.add_argument("-wp","--wandb_project",type=str, default="DeepLearning")

parser.add_argument("-we","--wandb_entity", type=str, default="cs22m020")

parser.add_argument("-d","--dataset", type=str,choices=["fashion_mnist","mnist"],  default="fashion_mnist")

parser.add_argument("-e","--epochs",type=int,default=10)

parser.add_argument("-b","--batch_size",type=int,default=64)

parser.add_argument("-l","--loss",choices=["mean_square", "cross_entropy"],default="cross_entropy")

parser.add_argument("-o","--optimizer",choices=["batch","momentum","nestrov","rmsProp","adam","Nadam"],default="Nadam")

parser.add_argument("-lr","--learning_rate",type=float,default=0.01)

parser.add_argument("-a","--activation",choices=["sigmoid", "tanh", "relu"],default="sigmoid")

parser.add_argument("-sz","--hidden_size",type=int,default=128)

parser.add_argument("-nhl","--num_layers",type=int,default=2)

parser.add_argument("-w_i", "--weight_init",choices=["random", "xavier"],default="xavier")

parser.add_argument("-eps", "--epsilon", type=float , default = 1e-10)

parser.add_argument("-beta2","--beta2",type=float, default=0.999)

parser.add_argument("-beta1","--beta1",type=float, default=0.9)

parser.add_argument("-beta","--beta",type=float, default=0.9)

parser.add_argument("-w_d","--weight_decay",type=float, default=0.0001)

parser.add_argument("-m","--momentum",type=float, default=0.9)

args = parser.parse_args()

wandb_project = args.wandb_project

wandb_entity = args.wandb_entity

iter=args.epochs

batchSize=args.batch_size

optimizer=args.optimizer

n=args.learning_rate

activation=args.activation

no_of_hidden_layers=args.num_layers

no_of_neuron=args.hidden_size

initialization=args.weight_init

input_neuron=784

alpha=args.weight_decay

loss_fn=args.loss

no_of_classes=10

epsilon=args.epsilon

beta=args.beta

beta1=args.beta1

beta2=args.beta2

momentum=args.momentum

dataset=args.dataset

run=wandb.init(project=wandb_project,entity=wandb_entity,reinit='true')





##########################################################    END       #########################################################


##########################################################  WandB Sweep ##########################################################





# default_params=dict(
#     iter=10,
#     batchSize=64,
#     optimizer='nadam',
#     n=0.01,
#     activation='sigmoid',
#     no_of_hidden_layers=2,
#     no_of_neuron=128,
#     initialization='xavier',
#     input_neuron=784, 
#     alpha=0.0001,
#     loss_fn='cross_entropy'
# )

# no_of_classes=10
# epsilon=1e-10
# beta=0.9
# beta1=0.9
# beta2=0.999
# momentum=0.9
# dataset="fashion_mnist"

# run=wandb.init(config=default_params,project='Deep Learning',entity='cs22m020',reinit='true')
# config=wandb.config

# iter=config.iter
# batchSize=config.batchSize
# optimizer=config.optimizer
# n=config.n
# activation=config.activation
# no_of_hidden_layers=config.no_of_hidden_layers
# no_of_neuron=config.no_of_neuron
# initialization=config.initialization
# input_neuron=config.input_neuron
# alpha=config.alpha
# loss_fn=config.loss_fn

# run.name='hl_'+str(no_of_hidden_layers)+'_bs_'+str(batchSize)+'_ac_'+str(activation)+' iter '+str(iter)+' neuron '+str(no_of_neuron)+' opt '+str(optimizer)+ ' eta '+str(n) +' alpha '+str(alpha)+' ini '+str(initialization)






##################################################     WandB and Sweep End    ###############################################



#Creating Placeholders 
x_train=None
x_test=None
y_train=None
y_test=None

#Loading the dataset
if dataset=="fashion_mnist":
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif dataset=="mnist":
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshaping and normalizing the dataset
x_train=x_train.reshape(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))
x_test=x_test.reshape(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))
x_train=x_train/255
x_test=x_test/255

#create validation dataset
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
            y=np.copy(z)
            for i in range(z.shape[0]):
                sum=0
                maxi=np.argmax(z[i])
                for j in range(z.shape[1]):
                    sum=sum+np.exp(z[i][j]-z[i][maxi])
                z[i]=np.exp(z[i]-z[i][maxi])/sum
                y[i]=z[i]
            return y
    
    def activations_derivative(self,activation,z):
        if activation=='sigmoid':
            return np.multiply((1/(1+np.exp(-z))),(1-(1/(1+np.exp(-z)))))
        elif activation=='relu':
            relu_derivative=np.maximum(0,z)
            relu_derivative[relu_derivative>0]=1
            return relu_derivative
        elif activation=='tanh':
            y=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
            return 1-np.square(y)
        
    def loss_function(self,loss_fn,yhat,y_train,alpha):
        reg=0
        for i in range(len(self.w)):
            reg=reg+np.sum(self.w[i]**2)
        reg=((alpha*reg)/2)

        if loss_fn=='cross_entropy':
            loss=0
            for i in range(y_train.shape[0]):
                loss+=-((np.log2(yhat[i][y_train[i]])))
            return (loss+reg)/y_train.shape[0]
        
        if loss_fn=='mean_square':
            el=np.zeros((y_train.shape[0],yhat.shape[1]))
            for i in range (y_train.shape[0]):
                el[i][y_train[i]]=1

            return ((np.sum(((yhat-el)**2)))+reg)/(y_train.shape[0])

    def make_layers(self,no_of_hidden_layers,no_of_neuron,input_neuron,initialization,no_of_classes):
        np.random.seed(5)
        layer=[]
        layer.append(input_neuron)
        for i in range(no_of_hidden_layers):
            layer.append(no_of_neuron)
        layer.append(no_of_classes)
        if initialization=='random':
            for i in range(no_of_hidden_layers+1):
                weights=np.random.uniform(-0.5,0.5,(layer[i],layer[i+1]))
                bias=np.random.uniform(-0.5,0.5,(1,layer[i+1]))
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
            if activation=='relu' and i==0:
                for i in range(a1.shape[0]):
                    maxi=np.argmax(a1[i])
                    a1[i]=a1[i]/a1[i][maxi]
            h1=self.activations(activation,a1)
            self.a.append(a1)
            self.h.append(h1)
            temp=h1
        a1=np.add(np.matmul(temp,self.w[l-1]),self.b[l-1])
        h1=self.activations('softmax',a1)
        self.a.append(a1)
        self.h.append(h1)
    
    def backward_pass(self,yhat,y_train,x_train,no_of_classes,activation,loss_fn,alpha):
        self.wd=[]
        self.bd=[]
        self.ad=[]
        self.hd=[]
        el=np.zeros((y_train.shape[0],no_of_classes))
        for i in range (y_train.shape[0]):
            el[i][y_train[i]]=1

        hd1=None
        ad1=None

        if loss_fn=="cross_entropy":
            yhatl=np.zeros((yhat.shape[0],1))
            for i in range (yhat.shape[0]):
                yhatl[i]=yhat[i][y_train[i]]

            hd1=-(el/yhatl)
            ad1=-(el-yhat)
            self.hd.append(hd1)
            self.ad.append(ad1)

        if loss_fn=="mean_square":
            hd1=2*(yhat-el)
            self.hd.append(hd1)
            ad1=[]
            for j in range(yhat.shape[1]):
                one_hot_j=np.zeros((yhat.shape[0],yhat.shape[1]))
                one_hot_j[:,j]=1
                yhat_j=np.ones((yhat.shape[0],yhat.shape[1]))*(yhat[:,j].reshape(yhat.shape[0],1))
                daj=2*(np.sum((yhat-el)*(yhat*(one_hot_j-yhat_j)),axis=1))
                ad1.append(daj)
            self.ad.append(np.array(ad1).T)


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
        
        for i in range(len(self.w)):
            self.wd[len(self.w)-1-i]=self.wd[len(self.w)-1-i]-alpha*self.w[i]
        
    
    def accuracy(self,x_test,y_test,activation):
        self.forward_pass(x_test,activation)
        l=len(self.w)
        ypred=np.argmax(self.h[l-1],axis=1)
        count=0
        for i in range(y_test.shape[0]):
            if ypred[i]!=y_test[i]:
                count=count+1
        return ((x_test.shape[0]-count)/y_test.shape[0])
    
    ############################################################################################################################################################

    def predict(self,x_test,y_test,activation):
        self.forward_pass(x_test,activation)
        l=len(self.w)
        ypred=np.argmax(self.h[l-1],axis=1)
        count=0
        for i in range(y_test.shape[0]):
            if ypred[i]!=y_test[i]:
                count=count+1

        
        #confusion matrix
        labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(preds=ypred, y_true=y_test,class_names=labels)})

        print("Test Accuracy: "+str(((x_test.shape[0]-count)/y_test.shape[0])))
    
    ###########################################################################################################################################################

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
    
    def onePass(self,x_train,y_train,no_of_classes,l,n,activation,loss_fn,alpha):
        self.forward_pass(x_train,activation)
        self.backward_pass(self.h[l-1],y_train,x_train,no_of_classes,activation,loss_fn,alpha)
    
    def batch(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,activation,loss_fn,alpha):
        data,ans=self.createBatches(x_train,y_train,batchSize)

        for i in range(iter):
            h=None
            for j in range(len(data)):
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation,loss_fn,alpha)
                for k in range (l):
                    self.w[k]=self.w[k]-n*(self.wd[l-1-k])
                    self.b[k]=self.b[k]-n*self.bd[l-1-k]

            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train,alpha)
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val,alpha)
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            print("Iteration Number: "+str(i)+" Train Loss : "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validation Loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            

    def momentum(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn,alpha):
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
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation,loss_fn,alpha)
                for k in range (l):
                    moment[k]=(moment[k]*beta)+self.wd[l-1-k]
                    momentB[k]=(momentB[k]*beta)+self.bd[l-1-k]
                    self.w[k]=self.w[k]-n*moment[k]
                    self.b[k]=self.b[k]-n*momentB[k]
  
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train,alpha)
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val,alpha)
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            print("Iteration Number: "+str(i)+" Train Loss : "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validation Loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def nestrov(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn,alpha):
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
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation,loss_fn,alpha)
                for k in range (l):
                    moment[k]=(beta*moment[k])+n*(self.wd[l-1-k])
                    momentB[k]=(beta*momentB[k])+n*self.bd[l-1-k]
                    self.w[k]=self.w[k]-moment[k]
                    self.b[k]=self.b[k]-momentB[k]
      
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train,alpha)
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val,alpha)
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            print("Iteration Number: "+str(i)+" Train Loss : "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validation Loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def rmsProp(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn,alpha,epsilon):
        data,ans=self.createBatches(x_train,y_train,batchSize)

        momentW=[]
        momentB=[]
        for i in range(l):
            temp=np.zeros((self.w[i].shape))
            temp2=np.zeros((self.b[i].shape))
            momentW.append(temp)
            momentB.append(temp2)

        for i in range(int(iter)):
            for j in range(len(data)):
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation,loss_fn,alpha)
                for k in range (l):
                    momentW[k]=(momentW[k]*beta)+(1-beta)*np.square(self.wd[l-1-k])
                    momentB[k]=(momentB[k]*beta)+(1-beta)*np.square(self.bd[l-1-k])
                    self.w[k]=self.w[k]-(n/np.sqrt(np.linalg.norm(momentW[k]+epsilon)))*self.wd[l-1-k]
                    self.b[k]=self.b[k]-(n/np.sqrt(np.linalg.norm(momentB[k]+epsilon)))*self.bd[l-1-k]
  
            self.forward_pass(x_train,activation)
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train,alpha)
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val,alpha)
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            print("Iteration Number: "+str(i)+" Train Loss : "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validation Loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def adam(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta1,beta2,activation,loss_fn,epsilon,alpha):
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

        t=0
        for i in range(int(iter)):
            for j in range(len(data)):
                t=t+1
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation,loss_fn,alpha)
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
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train,alpha)
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val,alpha)
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def Nadam(self,x_train,y_train,no_of_classes,l,iter,n,batchSize,beta1,beta2,activation,loss_fn,epsilon,alpha):
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

        t=0
        for i in range(int(iter)):
            for j in range(len(data)):
                t=t+1
                self.onePass(data[j],ans[j],no_of_classes,l,n,activation,loss_fn,alpha)
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
            loss_train=self.loss_function(loss_fn,self.h[l-1],y_train,alpha)
            self.forward_pass(x_val,activation)
            loss_val=self.loss_function(loss_fn,self.h[l-1],y_val,alpha)
            acc_train=self.accuracy(x_train,y_train,activation)
            acc_val=self.accuracy(x_val,y_val,activation)
            wandb.log({"train_accuracy":acc_train,"train_error":loss_train,"val_accuracy":acc_val,"val_error":loss_val})
            print("Iteration Number: "+str(i)+" Train Loss : "+str(loss_train))
            print("Iteration Number: "+str(i)+" Validation Loss : "+str(loss_val))
            print("Iteration Number: "+str(i)+" Train Accurcy : "+str(acc_train))
            print("Iteration Number: "+str(i)+" Validaion Accuracy: "+str(acc_val))
            
    def architecture(self,x_train,y_train,x_val,y_val,no_of_classes,no_of_hidden_layers,no_of_neuron,input_neuron,batchSize,initialization,loss_fn,activation,optimizer,n,iter,beta,beta1,beta2,epsilon,alpha,momentum):
        self.w=[]
        self.b=[]
        self.make_layers(no_of_hidden_layers,no_of_neuron,input_neuron,initialization,no_of_classes)
        l=len(self.w)
        if optimizer=="batch":
            self.batch(x_train,y_train,no_of_classes,l,iter,n,batchSize,activation,loss_fn,alpha)
        if optimizer=='momentum':
            self.momentum(x_train,y_train,no_of_classes,l,iter,n,batchSize,momentum,activation,loss_fn,alpha)
        if optimizer=='nestrov':
            self.nestrov(x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn,alpha)
        if optimizer=='rmsProp':
            self.rmsProp(x_train,y_train,no_of_classes,l,iter,n,batchSize,beta,activation,loss_fn,alpha,epsilon)
        if optimizer=='adam':
            self.adam(x_train,y_train,no_of_classes,l,iter,n,batchSize,beta1,beta2,activation,loss_fn,epsilon,alpha)
        if optimizer=='Nadam':
            self.Nadam(x_train,y_train,no_of_classes,l,iter,n,batchSize,beta1,beta2,activation,loss_fn,epsilon,alpha)


obj=NeuralNetwork()

obj.architecture(x_train,y_train,x_val,y_val,no_of_classes,no_of_hidden_layers,no_of_neuron,input_neuron,batchSize,initialization,loss_fn,activation,optimizer,n,iter,beta,beta1,beta2,epsilon,alpha,momentum)
obj.predict(x_test,y_test,activation)