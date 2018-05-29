
# coding: utf-8

# In[127]:


import selectivesearch
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.misc import imresize
import os
from six.moves import cPickle as pickle
from misc import *




CLASS_NAME = [
    'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
    'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc',
    'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks',
    'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'Background'
]




def get_bg(img,annot):
    proposals = get_region_proposals(img)
    bg = None
    for region in proposals:
        x,y,w,h = region
        box2 = [x,y,w+x,h+y]
        if iou(annot,box2) < 0.5:
            bg = box2
            break
    return bg



def crop_images(filename,istrain=True):
    with open(filename,"r") as f:
        for line in f:
            name = line.split()
            img = skimage.io.imread('data/images/'+name[0])
            x1,y1,x2,y2 = list(map(int,name[3:7]))
            try:
                img = img[y1:y2,x1:x2,:]
                img = imresize(img,(32,64,3),interp='bicubic')
                if istrain:
                    os.makedirs('data/train/'+name[1]+'/', exist_ok=True)
                    skimage.io.imsave('data/train/'+name[1]+'/'+name[0],img)
                else:
                    os.makedirs('data/test/'+name[1]+'/', exist_ok=True)
                    skimage.io.imsave('data/test/'+name[1]+'/'+name[0],img)

            except:
                pass




def get_dataset(path):
    dataset = []
    labels = []
    for i,name in enumerate(CLASS_NAME):
        direct_name = path+name+'/'
        direct = os.listdir(direct_name)
        for file in direct:
            img = skimage.io.imread(direct_name+file)
            img = scale(img)
            dataset.append(img)
            labels.append(i)
    labels = np.array(labels)
    dataset = np.array(dataset)
    return dataset,labels



def random_shuffle(dataset,labels):
    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm,:,:]
    labels = labels[perm]
    return dataset,labels




def save_pickle(filename):
    train_dataset, train_labels = get_dataset('data/train/')
    train_dataset, train_labels = random_shuffle(train_dataset,train_labels)
    test_dataset, test_labels = get_dataset('data/train/')
    test_dataset, test_labels = random_shuffle(test_dataset,test_labels)
    val_dataset = test_dataset[0:test_labels.shape[0]//2,:,:]
    val_labels = test_labels[0:test_labels.shape[0]//2]
    test_dataset = test_dataset[test_labels.shape[0]//2:,:,:]
    test_labels = test_labels[test_labels.shape[0]//2:]
    f = open(filename, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': val_dataset,
        'valid_labels': val_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()


# In[129]:


def main():

    #Uncomment this part if you want to generate a list of background images. 
    '''
    fp = open("data/flickr_logos_27_dataset_training_set_annotation.txt","r")

    lines = fp.readlines()


    write_lines = []
    for line in lines:
        name = line.split()
        img_name = name[0]
        img = skimage.io.imread('data/images/'+name[0])
        box1= list(map(int,name[3:7]))
        bg = get_bg(img,box1)
        write_lines.append(line)
        try:
            box2 = bg
            write_lines.append(name[0]+" Background "+str(27)+ " " +str(box2[0])+" " +str(box2[1])+" "+str(box2[2])+" "+str(box2[3]) +" \n")
        except:
            print("hi")
            pass


    
    write_lines = np.array(write_lines)
    np.random.shuffle(write_lines)

    train_list = write_lines[0:int(0.75*write_lines.shape[0])]
    test_list = write_lines[int(0.75*write_lines.shape[0]):]



    train_file = open("data/annot_with_bg_train.txt","w")
    train_file.writelines(list(train_list))
    train_file.close()
    test_file = open("data/annot_with_bg_test.txt","w")
    test_file.writelines(list(test_list))
    test_file.close()
    '''

    crop_images("data/annot_with_bg_train.txt")



    crop_images("data/annot_with_bg_test.txt",False)

    save_pickle('logo.pickle')

if __name__ == '__main__':
    main()

