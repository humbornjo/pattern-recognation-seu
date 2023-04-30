import numpy as np
from struct import unpack
import gzip,math,random,copy
import matplotlib.pyplot as plt

def read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img

def read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
    return lab

def normalize_image(img):
    mean=np.mean(img)
    std=np.sum((img-mean)**2)/(img.shape[0]*img.shape[1])
    return (img-mean)/std

def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

class Linear:
    def __init__(self,shape,lr=0.003):
        self.shape=shape
        self.lr=lr
        self.weight=np.random.normal(0, 1, (self.shape[0],self.shape[1]))
        self.bias=np.random.normal(0, 1, (1,self.shape[1]))

    def __call__(self,x):
        self.input=x
        self.output=np.matmul(x,self.weight)+self.bias
        return self.output

    def __backward__(self):
        return [self.input.transpose(),np.identity(self.shape[1])]

    def update(self,par):
        par_w=np.matmul(self.__backward__()[0],par)
        par_b=np.matmul(par,self.__backward__()[1])
        self.weight-=self.lr*par_w.astype(np.float)
        self.bias-=self.lr*par_b

class Softmax:
    def __init__(self,x,y):
        self.input = x.flatten()
        self.input-=self.input.max()
        self.label= y
        dnome=0
        for data in self.input:
            dnome+=math.pow(math.e,data)
        self.output=np.array([math.pow(math.e,ele)/dnome for ele in self.input])

    def __call__(self, *args, **kwargs):
        return self.output

    def __backward__(self):
        return np.array([self.output-self.label])


train_data_dir = './线性分类器分类任务/train-images-idx3-ubyte.gz'
train_label_dir = './线性分类器分类任务/train-labels-idx1-ubyte.gz'
test_data_dir = './线性分类器分类任务/t10k-images-idx3-ubyte.gz'
oh_train_label = one_hot_label(read_label(train_label_dir))
train_data = normalize_image(read_image(train_data_dir))
test_data = normalize_image(read_image(test_data_dir))


#************task 1************#
my_iter = [[train_data[i], oh_train_label[i]] for i in range(len(train_data))]
dataloader=copy.deepcopy(my_iter)
epoch=20
l=Linear([784,10])
res=[]
for i in range(epoch):
    random.shuffle(dataloader)
    for x,y in dataloader:
        l_out=l(np.array([x]))
        l.update(Softmax(l_out,y).__backward__())
    count, total = 0, 0
    for x, y in my_iter:
        total += 1
        count += l(np.array([x])).argmax() == np.array(y).argmax()
    res.append(count / total)
x=range(1,epoch+1,1)

plt.plot(x,res)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim(1,epoch)
plt.ylim(0,1)
plt.show()



#************task 2************#

#------------PCA-------------#
pca_mean=np.mean(train_data,axis=0)

temp = train_data - pca_mean
S=np.cov(temp.T)

eigenvalue,featurevector=np.linalg.eig(S)
#print(eigenvalue)
pca_num=80  ## PCA降到的维度
pca_data=np.matmul(train_data-pca_mean,featurevector[:,:pca_num])
#----------------------------#

##想使用PCA注释下面这行，取消注释上面这行

#pca_data=train_data

#------------LDA-------------#
by_class=[[] for _ in range(10)]
train_label = read_label(train_label_dir)
globle_mean=np.mean(pca_data,axis=0)
by_mean=[]
for i in range(len(pca_data)):
    by_class[train_label[i]].append(pca_data[i])
by_class=np.array(by_class)
for clas in by_class:
    by_mean.append(np.mean(clas,axis=0))

S_w=None
for i in range(10):
    temp = np.array(by_class[i]) - by_mean[i]
    try:
        S_w+=np.cov(temp.T)
    except:
        S_w=np.cov(temp.T)

S_b=None
for i in range(10):
    temp = (np.array(by_mean[i]) - globle_mean)[None]
    try:
        S_b+=len(by_class[i])*np.matmul(temp.T,temp)
    except:
        S_b=len(by_class[i])*np.matmul(temp.T,temp)

##寻找系数
'''
scale=10
flag=True
while flag==True:
    for i in range(1,10):
        try:
            np.linalg.inv(S_w + i*np.identity(784)/scale)
            print(i,scale)
            flag=False
            break
        except:
            continue
    scale*=10
'''

S_w_inverse=np.linalg.inv(S_w)
eigenvalue,featurevector=np.linalg.eig(np.matmul(S_w_inverse,S_b))

fnum=20  ## LDA降到的维度
lda_train_data=np.matmul(pca_data,featurevector[:,:fnum].astype(np.float64))
#----------------------------#


my_iter_lda = [[lda_train_data[i], oh_train_label[i]] for i in range(len(lda_train_data))]
dataloader_lda=copy.deepcopy(my_iter_lda)
epoch=20
l=Linear([fnum,10])
res=[]
for i in range(epoch):
    random.shuffle(dataloader_lda)
    for x,y in dataloader_lda:
        l_out=l(np.array([x]))
        l.update(Softmax(l_out,y).__backward__())
    count, total = 0, 0
    for x, y in my_iter_lda:
        total += 1
        count += l(np.array([x])).argmax() == np.array(y).argmax()
    res.append(count/total)

x=range(1,epoch+1,1)

plt.plot(x,res)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim(1,epoch)
plt.ylim(0,1)
plt.show()


