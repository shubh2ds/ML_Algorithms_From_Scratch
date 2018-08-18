from sklearn import datasets
import numpy as np
database=datasets.load_iris()
#print("Database is ",database)
total_index=np.random.permutation(150)
train_index=total_index[:120]
test_index=total_index[120:]

total_data=database.data
train_data=total_data[train_index]
test_data=total_data[test_index]

label=database.target
train_label=label[train_index]
test_label=label[test_index]
def distance(a,b):
    d=np.sqrt(np.square(a[0]-b[0])+np.square(a[1]-b[1])+np.square(a[2]-b[2])+np.square(a[3]-b[3]))
    return(d)

def find_neighbor(train_data,train_label,test_point,k=3):
    output=[]
    for i in range(len(train_data)):
        d=distance(train_data[i],test_point)
        index=i
        label=train_label[i]
        output.append((d,index,label))
    output.sort()
    neigh=[]
    for i in range(k):
        neigh.append(output[i])
    return(neigh)

def predict_label(neigh,k=3):
    count=[0,0,0]
    for i in range(k):
        if(neigh[i][2]==0):
            count[0]+=1
        if(neigh[i][2]==1):
            count[1]+=1
        if(neigh[i][2]==2):
            count[2]+=1
    
    predicted_label=count.index(max(count))
    return predicted_label

print("________________________________________")
pred_label=[] 
for i in range(30):
    nei=find_neighbor(train_data,train_label,test_data[i])
    pred_label.append(predict_label(nei))
print(nei)
print("predicted_label=",pred_label)
print("actual_label    ",test_label)
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(train_data[:,0],train_label,c='r')
plt.scatter(test_data[:,0],test_label,c="g")


def accuracy(predicted_label,test_label):
    c=0
    for i in range(30):
        if predicted_label[i]==test_label[i]:
            c+=1
        return(c)
from sklearn import metrics
accuracy=metrics.accuracy_score(pred_label,test_label)
#c1=int(accuracy(pred_label,test_label))
#accuracy=float((c1/30.0)*100)

#____________________________________________________________visualization_________________________

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_label,pred_label)
print("acc:",accuracy)
print("confusion_matrix:",cm)

plt.show()


