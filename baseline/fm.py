## which applies the implementation of FM from https://github.com/coreylynch/pyFM on the target datasets

from pyfm import pylibfm
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

def entry(uid, vid):
    return {"u": str(uid), "v": str(vid)}

## the dataset now in a form of np.array with column num as 3
def rearrange(ds, v, train = True):
    x = [];
    y = [];
    for row in ds:
        x.append(entry(int(row[0]), int(row[1])));
        y.append(float(row[2]))
    x  = v.fit_transform(x) if train else v.transform(x);
    return x, np.array(y);
    
    
    

## the data loaded should be in the form of numpy data file 
def load_data(path):
    train_set = np.load(path + '.train'); 
    test_set = np.load(path + '.test');
    vectorizer = DictVectorizer();
    x_train, y_train = rearrange(train_set, vectorizer, True);
    x_test, y_test = rearrange(test_set, vectorizer, False);
    return x_train, y_train, x_test, y_test;







if __name__ == '__main__':
    PREFIX = "/Users/morino/Downloads/dataset/";
    names = [ 'ml-latest-small/ml', 'BX-CSV-Dump/bx', 'jester/jester'];
    teller = ["MovieLens", "BookCrossing", "Jester"];
    max_epoch = 100;
    fm = None;
    for i, name in enumerate(names):
        print("BEGIN {}".format(teller[i]));
        x_train, y_train, x_test, y_test = load_data(PREFIX + name);
        fm = pylibfm.FM(num_factors=10, num_iter=max_epoch, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
        fm.fit(x_train,y_train);
        preds = fm.predict(x_test)
        print("{} FM RMSE: {}".format(teller[i],np.sqrt(mean_squared_error(y_test,preds))));
        print("END {}".format(teller[i]));
        
        
        
        
        
    
