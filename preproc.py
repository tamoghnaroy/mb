import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class preproc:

    def __init__(self, train_file,target_col,ID_col = None, test_file = None):

        self.train_raw = pd.read_csv(train_file)
        self.target = self.train_raw[target_col]
        self.nTrain = self.train_raw.shape[0]

        self.train_raw.drop([target_col], axis = 1, inplace = True)

        if test_file is None:
            self.test_raw = self.train_raw
        else:
            self.test_raw = pd.read_csv(test_file)

        if ID_col is None:
            self.train_raw_ID = None
            self.test_raw_ID = None
        else:
            self.train_raw_ID = self.train_raw[ID_col]
            self.test_raw_ID = self.test_raw[ID_col]

            self.train_raw.drop([ID_col], axis = 1, inplace = True)
            self.test_raw.drop([ID_col], axis = 1, inplace = True)

        self.train_raw_cols = self.train_raw.dtypes

        self.train = self.train_raw
        self.test = self.test_raw

    def col_cast(self,df):

        pass


    def remove_null(self):

        df = self.train.select_dtypes(exclude = [object])
        cols = df.loc[:,df.var(axis = 0) == 0].columns.tolist()
        self.train.drop(cols,axis = 1,inplace = True)
        self.test.drop(cols,axis = 1,inplace = True)


    def remove_duplicate(self):
        df = self.train.T.drop_duplicates().T
        dup_cols = list(set(self.train_raw.columns.tolist()) - set(df.columns.tolist()))

        self.train = self.train.drop(dup_cols, axis = 1)
        self.test = self.test.drop(dup_cols, axis = 1)


    def dummify(self):

        merged = pd.concat([self.train,self.test], axis = 0)
        merged = pd.get_dummies(merged, merged.select_dtypes(include = [object]).columns.tolist())

        self.train = merged[:self.nTrain]
        self.test = merged[self.nTrain:]

    def cat_encoding(self):

        for c in self.train.columns:
            if self.train[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(self.train[c].values) + list(self.test[c].values))
                self.train[c] = lbl.transform(list(self.train[c].values))
                self.test[c] = lbl.transform(list(self.test[c].values))
