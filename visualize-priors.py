
import numpy as np
import pandas as pd

# Load priors [.npy]
labels = np.load('/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/generation_labels_maf_guhfi2.npy', allow_pickle=True)
classes = np.load('/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/generation_classes_maf_guhfi2.npy', allow_pickle=True)
seg_names = np.load('/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/segmentation_names_maf_guhfi2.npy', allow_pickle=True)
seg_labels = np.load('/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/segmentation_labels_maf_guhfi2.npy', allow_pickle=True)
# Print the .npy
print('Generation labels: ', labels)
print('Generation classes: ', classes)
print('Segmentation labels: ', seg_labels)
#print('Segmentation names: ', seg_names)

# For visualization purposes, combine the three priors/arrays into one pandas dataframe
df1 = pd.DataFrame({'Generation labels': labels, 'Generation classes': classes, 'Segmentation labels': seg_labels, 'Segmentation names': seg_names})
# Set options to display all columns and increase display width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df1)
