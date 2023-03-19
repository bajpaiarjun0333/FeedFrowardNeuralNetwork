#install wandb
!pip install wandb -qU

# Log in to your W&B account
import wandb
import os
from keras.datasets import fashion_mnist

os.environ['WAND_NOTEBOOK_NAME']='question1'
!wandb login b0bbea67b5b95cece4e781392ed3f568328da17e

(X_train,Y_train),(X_test,Y_test)=fashion_mnist.load_data()

wandb.init(project='Deep Learning',entity='cs22m020',name='question1')
categories=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
collection=[]
label=[]
for i in range(len(X_train)):
  if(len(label)==10):
    break
  if(categories[Y_train[i]] in label):
    continue
  else:
    collection.append(X_train[i])
    label.append(categories[Y_train[i]])

wandb.log({"Question 1-Sample Images on Fashion Mnist": [wandb.Image(img, caption=lbl) for img,lbl in zip(collection,label)]})