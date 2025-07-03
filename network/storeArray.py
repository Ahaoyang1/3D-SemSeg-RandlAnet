import random
from collections import deque
import numpy as np
import torch

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        # self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

        self.store = [None for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for i in range(self.total_num_classes):
            idx = torch.where(class_ids==i)
            item = items[idx[0],:]
            item = item.cpu()
            if self.store[i] == None:
                self.store[i] = item[-self.items_per_class:, :]
            else:
                self.store[i] = torch.concat([self.store[i],item],dim=0)
                self.store[i] = self.store[i][-self.items_per_class:, :]

    def retrieve(self, class_id):
        all_items=[]
        if class_id != -1:
            return self.store[class_id]
        else:
            all_items=self.store
            return all_items

    def reset(self):
        # tmp = torch.zeros((1,36))
        self.store = [None for i in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])


if __name__ == "__main__":
    store = Store(10, 3)
    a=torch.rand(1,36)
    idx=torch.tensor([1])
    store.add(a,idx)
    print(store.retrieve(-1))
    tmp=store.store[0]
    print(tmp)
    if tmp.shape[0]==0:
        print(111)
    # store.add(('a', 'b', 'c', 'd', 'e', 'f'), (1, 1, 9, 1, 0, 1))
    # # print(store.retrieve(-1))
    # print(store.retrieve1(-1))
    # store.add(('h',), (4,))
    # # print(store.retrieve(9))
    # print(store.retrieve1(-1))

    # tmp =torch.zeros((1,36))
    # print(tmp.shape)
    # feature = torch.rand(10, 36)
    # label = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 1])
    # # feature=feature.numpy()
    # # label=label.numpy()
    # # idx = np.argwhere(label==1)
    # # print(idx)
    # idx = torch.where(label==2)
    # print(idx[0])
    #
    # a=torch.tensor([1,2,3])
    # # lst=[a,a]
    # # res=torch.tensor(lst)
    # # print(res.shape)
    # print(a.data)