#####################################
# GANGARAM ARVIND SUDEWAD 20CS30017 #
# KANCHI MOHAN KRISHNA 20CS10030    #
# LAV JHARVAL 20CS30031             #
#####################################

import tensorflow as tf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from collections import defaultdict
import os
import sys

####################
# INPUT PARAMETERS #
####################

# supported elements (atomic numbers)
# H, C, N, O, F, P, S, Cl, Br, I
element_list = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
elementdict = {8:"O", 7:"N", 6:"C", 1:"H", \
               9:"F", 15:"P", 16:"S", 17:"Cl", \
               35:"Br", 53:"I"}

# maximum path length in atompairs-fingerprint
APLength = 4

# directory, containing the models
model_dir = os.getcwd()+'/'

#################
# READ MOLECULE #
#################

# Open the SDF file
suppl = Chem.SDMolSupplier('molecule_10.sdf')

# Loop through the molecules in the SDF file
for mol in suppl:
    if mol is not None:
        # Add hydrogens to the molecule
        mH = Chem.AddHs(mol)
        num_atoms = mH.GetNumAtoms()
        
        # Check for unknown elements
        curr_element_list = []
        for at in mH.GetAtoms():
            element = at.GetAtomicNum()
            if element not in element_list:
                sys.exit("Error: element %i not known" % element)
            curr_element_list.append(element)
        curr_element_list = set(curr_element_list)
    

##############################
# LOAD NEURAL NETWORK MODELS #
##############################

nn = defaultdict(list)
for element in curr_element_list:
  nn[element] = tf.keras.models.load_model('model_'+elementdict[element]+'.h5')

####################################
# PREDICT CHARGES FOR THE MOLECULE #
####################################

pred_q = [0]*num_atoms
# loop over the atoms
for i in range(num_atoms):
  # generate atom-centered AP fingerprint
  fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mH, maxLength=APLength, fromAtoms=[i])
  arr = np.zeros(1,)
  DataStructs.ConvertToNumpyArray(fp, arr)
  # get the prediction from the neural network
  element = mH.GetAtomWithIdx(i).GetAtomicNum()
  pred_q[i] = nn[element].predict(arr.reshape(1,-1))[0][0]

##########################
# WRITE CHARGES TO FILE #
##########################
f = open('example.charg', 'w')
f.write('# aid\tpred\n')
for i, pred in enumerate(pred_q):
  f.write('%i\t%10.6f\n' % (i, pred))
f.close()
