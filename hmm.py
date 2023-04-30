import numpy as np
import random


class Preprocsss:  #"./隐马尔科夫分词任务/RenMinData.txt_utf8"
    def __init__(self, dir):
        self.dir = dir

    def mk_vocab(self):
        vocab = {}
        count = 0
        for sentence in self.data:
            for letter in sentence:
                if letter not in vocab.keys():
                    vocab[letter] = count
                    count += 1
        self.vocab = vocab
        self.word_num=len(vocab.keys())

    def mk_label(self):
        f = open(self.dir, 'r', encoding='utf8')
        f.readline()
        cursor = f.readline()
        data = []
        label = []
        while cursor:
            temp=[]
            sentence=cursor.strip()
            check=sentence.replace(' ', '')
            if len(check)>1:
                words = sentence.split(' ')
                for word in words:
                    if len(word) == 1:
                        temp+=[3]
                    else:
                        temp+=[0] + (len(word) - 2) * [1] + [2]
                data.append(check)
                label.append(temp)
                cursor=f.readline()
            else:
                cursor=f.readline()


        f.close()
        self.data_num=len(data)
        self.label_num = len(label)
        self.data=data
        self.label=label

    def piror(self):
        total_num=0
        num_BMES = 4 * [0]
        piror_a = [4*[0] for _ in range(4)]
        piror_b = [self.word_num * [0] for _ in range(4)]
        for i in range(self.data_num):
            if len(self.data[i])>1:
                piror_b[self.label[i][0]][self.vocab[self.data[i][0]]] += 1
                num_BMES[self.label[i][0]] += 1
                start = self.label[i][0]
                total_num += len(self.data[i])
                for j in range(1, len(self.data[i])):
                    piror_b[self.label[i][j]][self.vocab[self.data[i][j]]] += 1
                    num_BMES[self.label[i][j]] += 1
                    piror_a[start][self.label[i][j]] += 1
                    start=self.label[i][j]
                piror_a[start][self.label[i][0]] += 1
            else:
                continue
        self.piror_a=np.divide(np.array(piror_a).T,np.array(num_BMES),where=np.array(num_BMES)!=0).T
        self.piror_b=np.divide(np.array(piror_b).T,np.array(num_BMES),where=np.array(num_BMES)!=0).T
        self.part_BMES=np.array(num_BMES)/total_num


    def process(self):
        self.mk_label()
        self.mk_vocab()
        self.piror()
        return self.piror_a,self.piror_b,self.part_BMES

class HMM:
    def __init__(self,a,b,pi,vocab):
        self.a=a
        self.b=b
        self.pi=pi
        self.vocab=vocab

    ##viertbi predict
    def __call__(self, sentence):
        delta=[]
        psi=[]
        psi.append(4*[0])
        delta.append(self.pi*np.array([self.b[i][self.vocab[sentence[0]]] for i in range(4)]))
        for i in range(1,len(sentence)):
            delta.append(np.max((delta[i-1]*self.a.T).T,axis=0)*np.array([self.b[ii][self.vocab[sentence[i]]] for ii in range(4)]))
            psi.append(np.argmax((delta[i-1]*self.a.T).T,axis=0))
        omega=[]
        omega.append(np.argmax(delta[-1]))
        for i in range(len(psi)-1):
            omega.insert(0,psi[-1-i][omega[-1-i]])
        res=''
        for i in range(len(sentence)):
            if omega[i]==2 or omega[i]==3:
                res+=(sentence[i]+' ')
            else:
                res+=sentence[i]
        omega=[['B','M','E','S'][i] for i in omega]
        return res

    def forward(self,sample):
        data,label=sample
        alpha=[]
        alpha.append(self.pi*np.array([self.b[i][self.vocab[data[0]]] for i in range(4)]))
        for i in range(1,len(label)):
            alpha.append(np.matmul(alpha[i-1],self.a)*np.array([self.b[ii][self.vocab[data[i]]] for ii in range(4)]))
        return alpha

    def backward(self,sample):
        data,label=sample
        beta=[]
        beta.append(np.ones(4))
        for i in range(0,len(label)-1):
            beta.insert(0,(np.sum(beta[-1-i]*np.array([self.b[ii][self.vocab[data[-i-1]]] for ii in range(4)])*self.a,axis=1)))
        return beta

    def update(self,sample):
        flag=False
        data, label = sample
        alpha, beta = self.forward(sample), self.backward(sample)
        gamma = []
        dict_b = {}
        store=None
        ##求gamma矩阵 T*c*c
        for i in range(0, len(label) - 1):
            nume = (alpha[i] * self.a.T).T * (
                    beta[i + 1] * np.array([self.b[ii][self.vocab[data[i + 1]]] for ii in range(4)]))
            gamma.append(np.divide(nume , np.sum(nume), where=np.sum(nume) != 0))
        self.gamma = gamma
        ##求更新参数
        # 先求self.pi
        pi = np.sum(gamma[0], axis=1)
        # 求self.a
        nume_a = np.sum(np.array(gamma), axis=0)
        denom = np.sum(nume_a, axis=1)
        a = np.divide(nume_a.T, denom, where=denom != 0).T
        # 再求self.b
        for i in range(len(label) - 1):
            temp = np.sum(gamma[i], axis=1)
            try:
                dict_b[self.vocab[data[i]]] += np.divide(temp, denom, where=denom != 0)
            except:
                dict_b[self.vocab[data[i]]] = np.divide(temp, denom, where=denom != 0)
        b = self.b.T
        for key in dict_b.keys():
            b[key] = dict_b[key]
        b = b.T

p=Preprocsss("./隐马尔科夫分词任务/RenMinData.txt_utf8")
a,b,pi=p.process()
my_iter = [[p.data[i], p.label[i]] for i in range(len(p.data))]
hmm=HMM(a,b,pi,p.vocab)
'''
epoch=1
count=0
for i in range(epoch):
    random.shuffle(my_iter)
    for i in range(len(my_iter)//10000):
        temp_a,temp_b,temp_pi=hmm.update(my_iter[i])
        print(temp_a,temp_pi)
        hmm.a,hmm.b,hmm.pi=temp_a,temp_b,temp_pi'''

#print(hmm.a,hmm.pi)
sentences=['今天的天气很好。','学习模式识别课程是有难度的事情。','我是东南大学的学生。']
for sen in sentences:
    print(hmm(sen))
