"""
Short script to create new priors for SynthSeg generative model

MAF, August 2024, marc.a.fortin@ntnu.no

"""
### Imports
import os.path
import pandas as pd
import numpy as np

### User-defined variables

path2savepriors = '/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/'
new_generation_labels_filename = 'generation_labels_maf_guhfi2.npy'
new_generation_classes_filename = 'generation_classes_maf_guhfi2.npy'
new_segmentation_labels_filename = 'segmentation_labels_maf_guhfi2_med257.npy'
new_segmentation_names_filename = 'segmentation_names_maf_guhfi2_med257.npy'


### Create each prior as described in '2-generation-explained.py' from SynthSeg

# List of ROIs in the label maps where you want to generate synthetic intensities (doesn't need to be in all label maps)
# Ordering rules: 1) Background has to be 1st. 2) non-sided labels. 3) Left labels. 4) Right labels.
new_generation_labels = np.array([0, 14, 15, 16, 24, 77, 257, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63])
non_sided_labels = 7 # The first sided is #2 (Left-Cerebral-White-Matter)

# This is the array where you specify whether you want to segment all labels or not. If you want to segment all labels, this variables is exactly the same as 'new_generation_labels'. If that's not the case,
# you can specify below a new array where you specify what label you want to segment the label from new_generation_labels as. For example, here it is exactly the same *except* for label 257, which I 'segment' as background (label 0).
# Thus, instead of 257, we can observe that the value has been changed to 0 so now the original label 257 will be associated with the background.
new_segmentation_labels =  np.array([0, 14, 15, 16, 24, 77, 257, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63]) #new_generation_labels

# Conceptually, the weirdest prior. See email thread with B. Billot for a more detailled explanation because it doesn't really fit with the following one given here.
# Similar tissues 'should be' regrouped in the same 'class' ("so that intensities of similar regions are sampled from the same Gaussian distribution" [whatever that means]).
# same length and order as generation_labels. Class values should be between 0 and K-1 (where K is the total number of classes)
# F. eks., 'Left-Lateral-Ventricle' & 'Left-Inf-Lat-Vent' (and same for the right hemisphere version) have the same class in SynthSeg's model, 'Left-Hippocampus' & 'Left-Amygdala' also (and same for the right hemisphere version).
# Since we have no reason to change it from the original, we will follow the same idea with only the labels mentioned on the previous line to be in the same class.
# right/left labels are associated with the same class.
new_generation_classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19, 20, 21, 22, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32]) #OG value: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30, 31])

# Create an array of strings with the name of each corresponding label in 'new_generation_labels'. Pretty sure it is not relevant in practice since we don't use SynthSeg's training, but for consistency, let's do it.
new_segmentation_names = np.array(['background', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'CSF', 'WM-hypointensities', 'ExtraCerebral', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                                    'Left-Cerebellum-Cortex', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-choroid-plexus', 'Right-Cerebral-White-Matter',
                                    'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                                    'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-choroid-plexus'])

# Sanity check before saving
print("Size of array: ", len(new_generation_labels))
print("Size of array: ", len(new_generation_classes))
print("Size of array: ", len(new_segmentation_names))
print("Size of array: ", len(new_segmentation_labels))

# For visualization purposes, combine the three priors/arrays into one pandas dataframe
df1 = pd.DataFrame({'Generation labels': new_generation_labels, 'Generation classes': new_generation_classes, 'Segmentation labels': new_segmentation_labels, 'Segmentation names': new_segmentation_names})
# Set options to display all columns and increase display width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df1)

# Save the newly created priors
np.save(os.path.join(path2savepriors, new_generation_labels_filename), new_generation_labels) # New generation labels
np.save(os.path.join(path2savepriors, new_generation_classes_filename), new_generation_classes) # New generation classes
np.save(os.path.join(path2savepriors, new_segmentation_names_filename), new_segmentation_names) # New segmentation names
np.save(os.path.join(path2savepriors, new_segmentation_labels_filename), new_segmentation_labels) # New segmentation names
