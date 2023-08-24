#####################################
# GANGARAM ARVIND SUDEWAD 20CS30017 #
# KANCHI MOHAN KRISHNA 20CS10030    #
# LAV JHARVAL 20CS30031             #
#####################################

#!/usr/bin/env python

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys, os
from collections import defaultdict
import joblib
# from joblib import dump, load
import math

def calculate_rmse(pred_charges, true_charges):
    n = len(true_charges)
    rmse = math.sqrt(sum([(pred_charges[i] - true_charges[i])**2 for i in range(n)]) / n)
    return rmse



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

#directory, containing the models
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

#############################
# LOAD RANDOM FOREST MODELS #
#############################

rf = defaultdict(list)
for element in curr_element_list:
  rf[element] = joblib.load(model_dir+elementdict[element]+'.model')

####################################
# PREDICT CHARGES FOR THE MOLECULE #
####################################

pred_q = [0]*num_atoms
sd_rf = [0]*num_atoms
# loop over the atoms
for i in range(num_atoms):
  # generate atom-centered AP fingerprint
  fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mH, maxLength=APLength, fromAtoms=[i])
  arr = np.zeros(1,)
  DataStructs.ConvertToNumpyArray(fp, arr)
  # get the prediction by each tree in the forest
  element = mH.GetAtomWithIdx(i).GetAtomicNum()
  per_tree_pred = [tree.predict(arr.reshape(1,-1)) for tree in (rf[element]).estimators_]
  # then average to get final predicted charge
  pred_q[i] = np.average(per_tree_pred)
  # and get the standard deviation, which will be used for correction
  sd_rf[i] = np.std(per_tree_pred)

#########################
# CORRECT EXCESS CHARGE #
#########################

corr_q = [0]*num_atoms
# calculate excess charge
deltaQ = sum(pred_q)- float(AllChem.GetFormalCharge(mH))
charge_abs = 0.0
for i in range(num_atoms):
  charge_abs += sd_rf[i] * abs(pred_q[i])
deltaQ /= charge_abs
# correct the partial charges
for i in range(num_atoms):
  corr_q[i] = pred_q[i] - abs(pred_q[i]) * sd_rf[i] * deltaQ
    
##########################
# WRITE CHARGES TO FILE #
##########################
f = open('example1.charg', 'w')
f.write('# aid\tpred\n')
for i, pred in enumerate(corr_q):
  f.write('%i\t%10.6f\n' % (i, pred))
f.close()

# ####################
# Y = []
# element_indices = defaultdict(list)
# counter = 0
# # loop over molecules in database
# m = Chem.MolFromMolFile('molecule_10.sdf', removeHs=False)    
# # loop over atoms in molecule
# for at in m.GetAtoms():                             
#   aid = at.GetIdx()             
#   # read reference partial charge
#   q = float(m.GetAtomWithIdx(aid).GetProp('molFileAlias'))
#   Y.append(q)

true_charges = []
suppl = Chem.SDMolSupplier('molecule_10.sdf')
for mol in suppl:
    if mol is not None:
        for atom in mol.GetAtoms():
            true_charges.append(atom.GetDoubleProp('molFileAlias'))

rmse = calculate_rmse(corr_q, true_charges)
print(f"RMSE = {rmse:.4f}")

