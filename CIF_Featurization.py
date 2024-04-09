# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:29:33 2024

@author: brand
"""

import pandas as pd
import os
import pymatgen.core as pmg
from matminer.featurizers.structure import JarvisCFID

jarvis = JarvisCFID()
cif_path = 'CIF_Files/'
CIF_Files = os.listdir(cif_path)

jarvis_features = []
for cif in CIF_Files:
    cif_struc = pmg.Structure.from_file(cif_path + cif)
    cif_feature = jarvis.featurize(cif_struc)
    jarvis_features.append(cif_feature)
    
df = pd.DataFrame(jarvis_features, index=CIF_Files)

df.to_csv('C2DB_X.csv', index_label= 'CIF_Files')


