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
import tensorflow as tf
from sklearn.metrics import r2_score


####################
# INPUT PARAMETERS #
####################

# set NN parameters
learning_rate = 0.001
num_epochs = 100
batch_size = 64

# define the neural network architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(80, input_shape=(2048,), activation='relu'),
  tf.keras.layers.Dense(80, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])


# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='mean_squared_error')



# supported elements (atomic numbers)
# H, C, N, O, F, P, S, Cl, Br, I
# element_list = [ 7, 8, 9, 15, 16, 17, 35, 53]
element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
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


##################################
# TRAINING AND TESTING OF MODELS #
##################################

r2_list = []
# loop over elements
for element in element_list:
  # split data into train and test sets
  indices = element_indices[element]
  np.random.shuffle(indices)
  num_train = int(len(indices) * 0.8)
  train_indices = indices[:num_train]
  test_indices = indices[num_train:]

  # train the model on the training set
  X_train = np.array([X[i] for i in train_indices])
  Y_train = np.array([Y[i] for i in train_indices])
  dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size)
  model.fit(dataset_train, epochs=num_epochs)
  model.save('model_'+elementdict[element]+'.h5')

  # evaluate the model on the test set
  X_test = np.array([X[i] for i in test_indices])
  Y_test = np.array([Y[i] for i in test_indices])

    # loop over elements
#   for element in element_list:
  model = tf.keras.models.load_model('model_'+elementdict[element]+'.h5')  
  # predict charges on test set
  Y_pred = model.predict(X_test)  
  # calculate R2 score
  r2= r2_score(Y_test, Y_pred)
  r2_list.append(r2)

# print results
for element, r2 in zip(element_list, r2_list):
  print(f"Element: {elementdict[element]}")
  print(f"R2 score: {r2:.3f}")

# Get the count of atoms for each element
element_counts = {}
for element in element_list:
    element_counts[element] = len(element_indices[element])

# Print the counts
for element in element_list:
    print(f"Element {elementdict[element]}: {element_counts[element]}")

import matplotlib.pyplot as plt

# Create lists for x-axis (element names) and y-axis (counts)
x_values = [elementdict[element] for element in element_list]
y_values = [element_counts[element] for element in element_list]

# Create the bar chart
plt.bar(x_values, y_values)

# Add labels and title
plt.xlabel("Element")
plt.ylabel("Count")
plt.title("Count of Atoms for Each Element")

# save the graph
# plt.savefig('element_counts.png')
# Display the chart
# plt.show()

# Create lists for x-axis (element names) and y-axis (counts)
x_values = [elementdict[element] for element in element_list]
y_values = r2_list

# Create the bar chart
plt.bar(x_values, y_values)

# Add labels and title
plt.xlabel("Element")
plt.ylabel("r2_score")
plt.title("r2_score of Atoms for Each Element")

#save the graph as png
# plt.savefig('r2_score.png')
# Display the chart
# plt.show()