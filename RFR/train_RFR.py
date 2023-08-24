#####################################
# GANGARAM ARVIND SUDEWAD 20CS30017 #
# KANCHI MOHAN KRISHNA 20CS10030    #
# LAV JHARVAL 20CS30031             #
#####################################
#!/usr/bin/env python

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import os, glob
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# from sklearn.externals import joblib
import joblib
import random

####################
# INPUT PARAMETERS #
####################

# set RF-parameter
n_estimators = 100
min_samples_split = 6
random_state = 0
n_jobs = -1
max_depth = 6
min_samples_leaf = 6

# supported elements (atomic numbers)
# H, C, N, O, F, P, S, Cl, Br, I
# element_list = [ 7, 8, 9, 15, 16, 17, 35, 53]
element_list = [1, 6, 7, 8, 9, 16, 17, 35,53]
elementdict = {8:"O", 7:"N", 6:"C", 1:"H", \
               9:"F", 15:"P", 16:"S", 17:"Cl", \
               35:"Br", 53:"I"}

# maximum path length in atompairs-fingerprint
APLength = 4 

# directories
root = os.getcwd()+'/sdfs/'
dbfolds=['ZINC_mols','ChEMBL_mols']

#############################
# GENERATION OF DESCRIPTORS #
#############################

X = []
Y = []
element_indices = defaultdict(list)
counter = 0
# loop over molecules in database
for bd in dbfolds:
  basedir = root+bd
  for files in sorted(glob.glob(basedir+'/*.sdf'))[:]:
    m = Chem.MolFromMolFile(files, removeHs=False)    
    # loop over atoms in molecule
    for at in m.GetAtoms():                             
      aid = at.GetIdx()             
      # generate atom-centered AP fingerprint
      fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(m, maxLength=APLength, fromAtoms=[aid])
      arr = np.zeros(1,)
      DataStructs.ConvertToNumpyArray(fp, arr)
      X.append(arr)
      # read reference partial charge
      q = float(m.GetAtomWithIdx(aid).GetProp('molFileAlias'))
      Y.append(q)
      # store element
      element_indices[at.GetAtomicNum()].append(counter)
      counter += 1
print('size of database: ',counter)

##########################
# TRAINING OF RFR MODELS #
##########################

r2_list = []
# loop over elements
for element in element_list:
  # split data into training and test set
  indices=element_indices[element]
  np.random.shuffle(indices)
  num_train = int(0.8*len(indices))
  train_indices = indices[:num_train]
  test_indices = indices[num_train:]
  # train RF model
  XX = [X[i] for i in train_indices]
  YY = [Y[i] for i in train_indices]
  rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,\
                             min_samples_split=min_samples_split,n_jobs=n_jobs,\
                             min_samples_leaf=min_samples_leaf)
  X_test = [X[i] for i in test_indices]
  Y_test = [Y[i] for i in test_indices]

  Y_pred=rf.fit(XX, YY).predict(X_test)
  r2=r2_score(Y_test, Y_pred)
  r2_list.append(r2)
  # rf.fit(XX, YY)
  # write the model to file
  joblib.dump(rf, elementdict[element]+'.model', compress=9)

#print r2_list
for element, r2 in zip(element_list, r2_list):
  print(f"Element: {elementdict[element]}")
  print(f"R2 score: {r2:.3f}")

import matplotlib.pyplot as plt

# Get the count of atoms for each element

# Create lists for x-axis (element names) and y-axis (counts)
x_values = [elementdict[element] for element in element_list]
y_values = r2_list

# Create the bar chart
plt.bar(x_values, y_values)

# Add labels and title
plt.xlabel("Element")
plt.ylabel("r2_score")
plt.title("r2_score of Atoms for Each Element")
# plt.savefig('r2_score.png')

# Display the chart
# plt.show()

#save the graph as png

# Get the count of atoms for each element
element_counts = {}
for element in element_list:
    element_counts[element] = len(element_indices[element])

# Print the counts
for element in element_list:
    print(f"Element {elementdict[element]}: {element_counts[element]}")

# Create lists for x-axis (element names) and y-axis (counts)
x_values = [elementdict[element] for element in element_list]
y_values = [element_counts[element] for element in element_list]

# Create the bar chart
plt.bar(x_values, y_values)

# Add labels and title
plt.xlabel("Element")
plt.ylabel("Count")
plt.title("Count of Atoms for Each Element")

#save the graph as png
# plt.savefig('atom_count.png')
# Display the chart
# plt.show()

