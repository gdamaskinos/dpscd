"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import random
random.seed(1)
np.random.seed(1)
import sys
import os
import shutil
import wget
import bz2

from sklearn.model_selection import train_test_split
from sklearn import datasets, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from os.path import expanduser
home = expanduser("~")

def load(name):
  if name == 'news20':
    return news20()
  elif name == 'news20small':
    return news20small()
  elif name == 'adult':
    return adult()
  elif name == 'phishing':
    return phishing()
  elif name == 'mushrooms':
    return mushrooms()
  elif name == 'breastcancer':
    return breast_cancer()
  elif name == 'higgs':
    return higgs()
  elif name == 'iris':
    return iris()
  elif name == 'epsilon':
    return epsilon()
  elif name == 'banking':
    return banking()
  elif name == 'diabetes':
    return diabetes()
  elif name == 'E2006small':
    return E2006small()
  elif name == 'E2006':
    return E2006()
  elif name == 'msd':
    return msd()
  elif name == 'cadata':
    return cadata()
  elif name == 'cpusmall':
    return cpusmall()
  elif name == 'abalone':
    return abalone()
  else:
    print("Dataset name not found", flush=True)

def iris():
  iris = datasets.load_iris()
  X = iris.data[:, :2]
  y = (iris.target != 0) * 1
  y[y == 0] = -1
  X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2)

  return X_train, Y_train, X_test, Y_test


def higgs_sample(data_path='dac_sample.txt'):
  """Splits higgs dataset sample given for PCML course"""

  data = pd.read_csv(data_path, sep='\t', header=None)#, nrows= 1000)
  features_col = np.arange(1,39)
  # data.dropna(axis='index',subset=features_col, inplace=True) #drop all lines with NaN values
  data.fillna(value=0, inplace=True) #fill all NaN values with 0
  numerical_features_col = np.arange(1,14)
  categorical_features_col = np.arange(14,39)
  features = data.iloc[:,1:39].copy()
  labels = data.iloc[:,0].copy()
  for col in categorical_features_col:
      features[col] = features[categorical_features_col][col].astype('category')
  cat_cols = features.select_dtypes(['category']).columns
  features[cat_cols] = features[cat_cols].apply(lambda col : col.cat.codes)

  # Create unbalanced dataset
  indexes = []
  ones = labels[labels == 1]
  zeros = labels[labels == 0]
  for i in range(2000):
      indexes.append(ones.index[i])
  for i in range(len(zeros)):
      indexes.append(zeros.index[i])
  len(indexes)
  features = features.iloc[indexes]
  labels = labels[indexes]

  # Normalize
  features = (features - features.mean()) / (features.max() - features.min())

  X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.33, random_state=4)
  X_train = X_train.values
  Y_train = Y_train.values
  X_test = X_test.values
  Y_test = Y_test.values
  Y_train[Y_train == 0] = -1
  Y_test[Y_test == 0]   = -1
  N, M = X_train.shape

  return X_train, Y_train, X_test, Y_test

def higgs(data_path=home+'/datasets/higgs/HIGGS'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download HIGGS.bz2 from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  X_train = preprocessing.normalize(X_train, axis=0, norm='l1')
  X_test = preprocessing.normalize(X_test, axis=0, norm='l1')

  return X_train, Y_train, X_test, Y_test

def phishing(data_path=home+'/datasets/phishing/phishing'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download phishing from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)

  X = X.toarray()

  Y[Y == 0] = -1

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test


def mushrooms(data_path=home+'/datasets/mushrooms/mushrooms'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download phishing from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)

  X = X.toarray()

  Y[Y == 2] = -1

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test


def epsilon(train_path=home+'/datasets/epsilon/epsilon_normalized',
        test_path=home+'/datasets/epsilon/epsilon_normalized.t'):

  try:
    open(train_path, 'r')
    open(test_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download epsilon from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html")
    return None, None, None, None

  X_train, Y_train = datasets.load_svmlight_file(train_path)
  X_test, Y_test = datasets.load_svmlight_file(test_path)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  return X_train, Y_train, X_test, Y_test

def banking(data_path=home+'/datasets/bank_marketing/banking.csv'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download from https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv")
    return None, None, None, None

  data = pd.read_csv(data_path, header=0)
  data = data.dropna()
  data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]],
            axis=1, inplace=True)

  data2 = pd.get_dummies(data, columns =['job', 'marital', 'default',
                                         'housing', 'loan', 'poutcome'])

  data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)

  X = data2.iloc[:,1:]
  y = data2.iloc[:,0]
  X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
  Y_train[Y_train == 0] = -1
  Y_test[Y_test == 0] = -1

  X_train = X_train.values
  Y_train = Y_train.values
  X_test = X_test.values
  Y_test = Y_test.values

  return X_train, Y_train, X_test, Y_test

  # normalization makes learning very slow (due to lack of sparsity)
#  X_train = preprocessing.normalize(X_train, axis=0, norm='l1')
#  X_test = preprocessing.normalize(X_test, axis=0, norm='l1')

  return X_train, Y_train, X_test, Y_test


def news20(data_path=home+'/datasets/news20/news20.binary'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download news20.binary from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)


  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  # normalization makes learning very slow (due to lack of sparsity)
  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  return X_train, Y_train, X_test, Y_test

def news20small(data_path=home+'/datasets/news20/news20small'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download news20.binary from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html and run: ```head -1300 news20.binary > news20small```")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)

  X = X.toarray()

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
      random_state=42)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test


def linearSample(N=50, variance=100):

  X = np.array(range(N)).T + 1
  Y = np.array([random.random() * variance + i * 10 + 900 for i in range(len(X))]).T

  X = X.reshape(-1,1)

  return X, Y, X, Y

def breast_cancer():
  data = datasets.load_breast_cancer()
  X = data.data
  Y = data.target

  Y[Y == 0] = -1

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
      random_state=42)

  X_train = preprocessing.scale(X_train)
  X_test = preprocessing.scale(X_test)
  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)


  return X_train, Y_train, X_test, Y_test

def diabetes():
  diabetes = datasets.load_diabetes()

  # Split the data into training/testing sets
  X_train = diabetes.data[:-20]
  X_test = diabetes.data[-20:]

  # Split the targets into training/testing sets
  Y_train = diabetes.target[:-20]
  Y_test = diabetes.target[-20:]

  X_train = preprocessing.scale(X_train)
  X_test = preprocessing.scale(X_test)

  return X_train, Y_train, X_test, Y_test

def cpusmall(data_path=home+'/datasets/cpusmall/cpusmall'):
  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download cpusmall from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  y_m = np.mean(Y_train)
  y_s = np.std(Y_train)

  Y_train = (Y_train-y_m)
  Y_test = (Y_test-y_m)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test

def abalone(data_path=home+'/datasets/abalone/abalone'):

  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download abalone from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  y_m = np.mean(Y_train)
  y_s = np.std(Y_train)

  Y_train = (Y_train-y_m)
  Y_test = (Y_test-y_m)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test


def cadata(data_path=home+'/datasets/cadata/cadata'):
  """Reported performance:
    http://www.jmlr.org/papers/volume18/15-025/15-025.pdf
    http://www.stat.cmu.edu/~cshalizi/350/hw/solutions/solutions-06.pdf
  """

  try:
    open(data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download cadata from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html")
    return None, None, None, None

  X, Y = datasets.load_svmlight_file(data_path)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  y_m = np.mean(Y_train)
  y_s = np.std(Y_train)

  Y_train = (Y_train-y_m)
  Y_test = (Y_test-y_m)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test

def msd(train_path=home+'/datasets/yearPredictionMSD/YearPredictionMSD',
        test_path=home+'/datasets/yearPredictionMSD/YearPredictionMSD.t'):

  try:
    open(train_path, 'r')
    open(test_path, 'r')
  except FileNotFoundError as e:
    print("Dataset not found in: " + home + "/datasets/yearPredictionMSD/")
    path = "/tmp/yearPredictionMSD/"
    print("Looking for dataset in: " + path)
    files = ["YearPredictionMSD.bz2", "YearPredictionMSD.t.bz2"]
    if not os.path.exists(path):
      os.mkdir(path)
      print("Downloading and extracting dataset in: " + path)
      url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/"
      for f in files:
        filepath = path + f
        wget.download(url + f, filepath)
        print("\nExtracting...")
        zipfile = bz2.BZ2File(filepath)
        data = zipfile.read()
        newfilepath = filepath[:-4]
        open(newfilepath, 'wb').write(data)
    train_path = path + files[0]
    test_path = path + files[0]
    print("Dataset is now in: " + path)

  X_train, Y_train = datasets.load_svmlight_file(train_path)
  X_test, Y_test = datasets.load_svmlight_file(test_path)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  y_m = np.mean(Y_train)
  y_s = np.std(Y_train)

  Y_train = (Y_train-y_m)
  Y_test = (Y_test-y_m)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test

def E2006(train_path=home+'/datasets/E2006/E2006.train',
    test_path=home+'/datasets/E2006/E2006.test'):
  """Download E2006-tfidf from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html
    Reported performance:
    [Table 6](http://www.kdd.org/kdd2016/papers/files/rpp0242-zhangA.pdf)
  """

  try:
    open(train_path, 'r')
    open(test_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download E2006-tfidf from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html")
    return None, None, None, None

  X_train, Y_train = datasets.load_svmlight_file(train_path)
  X_test, Y_test = datasets.load_svmlight_file(test_path)

  X_train = X_train.toarray()
  X_test = X_test.toarray()

  # X_test is missing two columns (all zeros)
  z = np.zeros((X_test.shape[0], 2))
  X_test = np.append(X_test, z, axis=1)

  y_m = np.mean(Y_train)
  y_s = np.std(Y_train)

  Y_train = (Y_train-y_m)
  Y_test = (Y_test-y_m)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  return X_train, Y_train, X_test, Y_test

def E2006small(data_path=home+'/datasets/E2006/E2006.small'):

  X, Y = datasets.load_svmlight_file(data_path)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
      random_state=42)

  return X_train.toarray(), Y_train, X_test.toarray(), Y_test


def adult(data_path=home+'/datasets/adult/'):
  """Preprocessing code fetched from
  https://github.com/animesh-agarwal/Machine-Learning-Datasets/tree/master/census-data"""
  train_data_path = os.path.join(data_path, 'adult.data')
  test_data_path = os.path.join(data_path, 'adult.test')
  try:
    open(train_data_path, 'r')
    open(test_data_path, 'r')
  except FileNotFoundError as e:
    print(str(e))
    print("Download `adult.data` and `adult.test` from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/")
    return None, None, None, None


  columns = ["age", "workClass", "fnlwgt", "education", "education-num",
             "marital-status", "occupation", "relationship", "race", "sex",
             "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

  train_data = pd.read_csv(train_data_path, names = columns, sep=' *, *', na_values='?')
  test_data = pd.read_csv(test_data_path, names = columns, sep=' *, *', skiprows =1, na_values='?')

  num_attributes = train_data.select_dtypes(include=['int'])
  cat_attributes = train_data.select_dtypes(include=['object'])

  class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, type):
      self.type = type

    def fit(self, X, y=None):
      return self

    def transform(self,X):
      return X.select_dtypes(include=[self.type])

  class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns = None, strategy='most_frequent'):
      self.columns = columns
      self.strategy = strategy


    def fit(self,X, y=None):
      if self.columns is None:
        self.columns = X.columns

      if self.strategy is 'most_frequent':
        self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
      else:
        self.fill ={column: '0' for column in self.columns}

      return self

    def transform(self,X):
      X_copy = X.copy()
      for column in self.columns:
        X_copy[column] = X_copy[column].fillna(self.fill[column])
      return X_copy

  class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropFirst=True):
      self.categories=dict()
      self.dropFirst=dropFirst

    def fit(self, X, y=None):
      join_df = pd.concat([train_data, test_data])
      join_df = join_df.select_dtypes(include=['object'])
      for column in join_df.columns:
        self.categories[column] = join_df[column].value_counts().index.tolist()
      return self

    def transform(self, X):
      X_copy = X.copy()
      X_copy = X_copy.select_dtypes(include=['object'])
      for column in X_copy.columns:
        X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
      return pd.get_dummies(X_copy, drop_first=self.dropFirst)

  num_pipeline = Pipeline(steps=[
      ("num_attr_selector", ColumnsSelector(type='int')),
      ("scaler", StandardScaler())
  ])

  cat_pipeline = Pipeline(steps=[
      ("cat_attr_selector", ColumnsSelector(type='object')),
      ("cat_imputer", CategoricalImputer(columns=['workClass','occupation', 'native-country'])),
      ("encoder", CategoricalEncoder(dropFirst=True))
  ])

  full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])

  train_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
  test_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)

  train_copy = train_data.copy()

  # convert the income column to 0 or 1 and then drop the column for the feature vectors
  train_copy["income"] = train_copy["income"].apply(lambda x:0 if x=='<=50K' else 1)

  X_train = train_copy.drop('income', axis =1)
  Y_train = train_copy['income']

  X_train = full_pipeline.fit_transform(X_train)
  test_copy = test_data.copy()

  # convert the income column to 0 or 1
  test_copy["income"] = test_copy["income"].apply(lambda x:0 if x=='<=50K.' else 1)

  # separating the feature vecotrs and the target values
  X_test = test_copy.drop('income', axis =1)
  Y_test = test_copy['income']

  # preprocess the test data using the full pipeline
  # here we set the type_df param to 'test'
  X_test = full_pipeline.fit_transform(X_test)

  s = preprocessing.MaxAbsScaler()
  X_train = s.fit_transform(X_train)
  X_test  = s.transform(X_test)

  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)

  Y_train = Y_train.to_numpy()
  Y_test = Y_test.to_numpy()

  Y_train[Y_train == 0] = -1
  Y_test[Y_test == 0] = -1

  return X_train, Y_train, X_test, Y_test
