
import numpy as np
import pandas as pd

# Load priors [.npy]
labels = np.load('/home/marcantf/Code/SynthSeg/data/labels_classes_priors/generation_labels.npy', allow_pickle=True)
classes = np.load('/home/marcantf/Code/SynthSeg/data/labels_classes_priors/generation_classes.npy', allow_pickle=True)
segs = np.load('/home/marcantf/Code/SynthSeg/data/labels_classes_priors/synthseg_segmentation_names.npy', allow_pickle=True)

# Print the .npy
print('Generation labels: ', labels)
print('Generation classes: ', classes)
print('Segmentation names: ', segs)

# For visualization purposes, combine the three priors/arrays into one pandas dataframe
df1 = pd.DataFrame({'Generation labels': labels, 'Generation classes': classes, 'Segmentation names': segs})
print(df1)
