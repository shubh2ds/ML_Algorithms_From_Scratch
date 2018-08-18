import numpy as np
import matplotlib.pyplot as plt
#data=np.random.randint(1,20,[10,2])

from sklearn import datasets
datasets=datasets.load_iris()
data=datasets.data
class kmeans:
    def __init__(self,k=3,max_iter=10):
        self.k=k
        self.max_iter=max_iter
    def fit(self,database):
        self.centroid={}
        for i in range(self.k):
            self.centroid[i]=database[i]
        print(self.centroid)
        for i in range(self.max_iter):
            self.classified_data={}
            
            for i in range(self.k):
                self.classified_data[i]=[]
            print(self.classified_data)
            for i in range(len(database)):
                d=[]
                for j in range(len(self.centroid)):
                    d.append(np.linalg.norm(database[i]-self.centroid[j]))
                    idx=d.index(min(d))
                    self.classified_data[idx].append(database[i])
            #print(self.classified_data)
            color=["g","c","b",]
            lab=["0:","1:","2:"]
            plt.ion()
            plt.cla()
            plt.figure(1)
            
        
            for label in (self.classified_data):
                col=color[label]
                lab1=lab[label]
                for points in (self.classified_data[label]):
                    plt.scatter(points[0],points[1],c=col,label=lab1)
                plt.scatter(self.centroid[label][0],self.centroid[label][1],s = 100,c="r", label=lab1)
                #plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
            previous_centroid=dict(self.centroid)
            for i in range(self.k):
                self.centroid[i]=np.average(self.classified_data[i],axis=0)
            plt.pause(1)
            plt.ioff()
            optimised=True
            for i in range(self.k):
                current_centroid=self.centroid[i];
                original_centroid=previous_centroid[i];
                d=np.linalg.norm(current_centroid-original_centroid)
                print(d)
                if(d>0.1):
                    optimised=False
            if(optimised==True):
                break;

clf=kmeans()
clf.fit(data)
