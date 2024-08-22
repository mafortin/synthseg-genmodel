"""
Short script to create new priors for SynthSeg generative model

MAF, August 2024, marc.a.fortin@ntnu.no

"""
import os.path

### Imports

import numpy as np

### User-defined variables

path2savepriors = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors'
new_generation_labels_filename = 'generation_labels_maf.npy'
new_generation_classes_filename = 'generation_classes_maf.npy'
new_segmentation_labels_filename = 'segmentation_labels_maf.npy'
new_segmentation_names_filename = 'segmentation_names_maf.npy'


### Create each prior as described in '2-generation-explained.py' from SynthSeg

# List of ROIs in the label maps where you want to generate synthetic intensities (doesn't need to be in all label maps)
# Ordering rules: 1) Background has to be 1st. 2) non-sided labels. 3) Left labels. 4) Right labels.
new_generation_labels = np.array([0, 14, 15, 16, 24, 77, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63])
non_sided_labels = 6 # The first non-sided is #2 (Left-Cerebral-White-Matter)

# Since we want to segment every label in 'new_generation_labels' we can just copy-paste the same array for the labels to be segmented.  Pretty sure it is not relevant in practice since we don't use SynthSeg's training, but for consistency, let's do it.
new_segmentation_labels = new_generation_labels # Same value as : new_generation_labels

# Conceptually, the weirdest prior. See email thread with B. Billot for a more detailled explanation because it doesn't really fit with the following one given here.
# Similar tissues 'should be' regrouped in the same 'class' ("so that intensities of similar regions are smpled from the same Gaussian distribution" [whatever it means]).
# same length and order as generation_labels. Class values should be between 0 and K-1 (where K is the total number of classes)
# F. eks., 'Left-Lateral-Ventricle' & 'Left-Inf-Lat-Vent' (and same for the right hemisphere version) have the same class in SynthSeg's model, 'Left-Hippocampus' & 'Left-Amygdala' also (and same for the right hemisphere version).
# Since we have no reason to change it from the original, we will follow the same idea with only the labels mentioned on the previous line to be in the same class.
# right/left labels are associated with the same class.
new_generation_classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 28, 29, 30, 31])

# Create an array of strings with the name of each corresponding label in 'new_generation_labels'. Pretty sure it is not relevant in practice since we don't use SynthSeg's training, but for consistency, let's do it.
new_segmentation_names = np.array(['background', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'CSF', 'WM-hypointensities', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
                                    'Left-Cerebellum-Cortex', 'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-choroid-plexus', 'Right-Cerebral-White-Matter',
                                    'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                                    'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-choroid-plexus'])

# Save the newly created priors

np.save(os.path.join(path2savepriors, new_generation_labels_filename), new_generation_labels) # New generation labels
np.save(os.path.join(path2savepriors, new_generation_classes_filename), new_generation_classes) # New generation classes
np.save(os.path.join(path2savepriors, new_segmentation_names_filename), new_segmentation_names) # New segmentation names
np.save(os.path.join(path2savepriors, new_segmentation_labels_filename), new_segmentation_labels) # New segmentation names
