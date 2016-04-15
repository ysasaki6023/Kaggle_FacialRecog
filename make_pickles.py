import numpy as np
import pandas as pd
import cPickle as pickle
import six

def readFile(inFile,outFile):
    df = pd.read_csv(inFile)
    #print df.columns
    #print df["Image"].head()
    #print df["left_eyebrow_outer_end_x"].head()
    dic = {}
    for i in df.columns:
        dic[i] = np.array(df[i])
        print i
        if i!="Image":
            print dic[i],dic[i].size
            #raw_input()
        else:
            p = np.zeros((df[i].size,96*96))
            for j,k in enumerate(df[i]):
                #if j>100: break
                k = np.array(k.split())
                k = k.astype(np.float)
                p[j] = k
                print j
            dic[i] = p
    #print dic
    #pickle.dump(dic,open(outFile,"wb"))
    six.moves.cPickle.dump(dic,open(outFile,"wb"),-1)




readFile("data/training.csv","data/training.pickles")
