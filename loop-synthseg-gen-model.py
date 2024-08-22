"""
Short script to loop the generative model of SynthSeg to create synthetic MR images from a dataset of ASEG.nii.gz segmentations.
This is basically a copied-pasted version of the original scripts from SynthSeg repository but adapted to my usage.

Author: MAF, Juli 2023 marc.a.fortin@ntnu.no

"""
### Imports

import os
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

### Inputs from user

# number of synthetic images created per subject
nsim = 4
output_shape = 300 # (vol_shape)3 of the synthetic images
# Paths
path2dataset = '/home/marcantf/Data/training/HCP_20/100408' # path to segmentations (i.e., inputs to generative model) . This needs to be the path to
# the directory containing individual folders for each
path2synthimg = '/home/marcantf/Data/training/synthetic/test' # path to synthetic MR images (i.e., output path)
one_folder = 1
# subjects.
hcp = 1 # if the dataset is from the HCP set this to 1, otherwise 0

# ---------- Generation parameters ----------
# These parameters are explained in SynthSeg-dir/scrips/tutorials/2-generation_explained.py.
# These are 'hard-coded' so you should not modify them except if you're optimizing the model.

n_channels = 1
n_neutral_labels = 6 # Mumber of non-sided labels in your segmentations. The value has probably been previously determined if you have created your own 'generation_labels.npy'
prior_distributions = 'uniform' # GMM sampling: 'uniform' eller 'normal'
# spatial deformation parameters
flipping = True
scaling_bounds = .15
rotation_bounds = 15
shearing_bounds = .012
translation_bounds = False
nonlin_std = 3.
bias_field_std = .7
# acquisition resolution parameters
randomise_res = False
# shape and resolution of the outputs
target_res = None
# Matrix Size of synthetic image created
output_shape = 300

# Paths to the generative model setup
path_generation_labels = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors/generation_labels_maf.npy' # generation labels
path_segmentation_labels = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors/segmentation_labels_maf.npy' # segmentation labels
path_generation_classes = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors/generation_classes_maf.npy' # classes generation

### Data preparation

direc = os.listdir(path2dataset)
fullpath2subdir = [path2dataset + x for x in direc]
path2sub = [s for s in fullpath2subdir if os.path.isdir(s) if not "md5" in s]
subslist = [x for x in direc if not "md5" in x if not "." in x]

if hcp:
    subsubdir = 'T1w'
    aseg_name = 'aseg_noCC.nii.gz' # !!! this naming convention is subject to modification !!!
    #fullpath2aseg = [os.path.join(f, subsubdir, aseg_name) for f in path2sub]
    fullpath2aseg = [os.path.join(path2dataset, subsubdir, aseg_name)]
#else: # to be coded



### Generate synthetic images for all subjects in the dataset

for idx, sub in enumerate(fullpath2aseg): #loop through all subjects

    print('Processing Subject %s!' % (idx))
    for n in range(nsim): #loop through the number of synthetic image you want to create from each subject

        print('Creating synthetic image #%s for subject #%s!' % (n, idx))
        # instantiate BrainGenerator object
        brain_generator = BrainGenerator(labels_dir=sub,
                                         generation_labels=path_generation_labels,
                                         n_neutral_labels=n_neutral_labels,
                                         prior_distributions=prior_distributions,
                                         generation_classes=path_generation_classes,
                                         output_labels=path_segmentation_labels,
                                         n_channels=n_channels,
                                         target_res=target_res,
                                         output_shape=output_shape,
                                         flipping=flipping,
                                         scaling_bounds=scaling_bounds,
                                         rotation_bounds=rotation_bounds,
                                         shearing_bounds=shearing_bounds,
                                         translation_bounds=translation_bounds,
                                         nonlin_std=nonlin_std,
                                         bias_field_std=bias_field_std,
                                         randomise_res=randomise_res)

        # generate new image and corresponding labels
        im, lab = brain_generator.generate_brain()

        # Reduce precision level of both the synthetic image and label map (from 32 to 16bit)
        im.astype(np.float16)
        lab.astype(np.int16)

        if one_folder:

            # Create folder if not already created
            fullpath2output = path2synthimg

            if not os.path.exists(fullpath2output):
                os.makedirs(fullpath2output)

            # save output image and label map
            utils.save_volume(im, brain_generator.aff, brain_generator.header,
                              os.path.join(fullpath2output, 'synth_img_%s_%s.nii.gz' % (idx, n)))
            utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                              os.path.join(fullpath2output, 'segm_%s_%s.nii.gz' % (idx, n)))

        #else:
        #    # Create folder if not already created
        #    fullpath2output = os.path.join(path2synthimg, subslist[idx])

        #    if not os.path.exists(fullpath2output):
        #        os.makedirs(fullpath2output)

            # save output image and label map
        #    utils.save_volume(im, brain_generator.aff, brain_generator.header,
        #                      os.path.join(fullpath2output,  'synth_img_%s.nii.gz' % (nsim)))
        #    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
        #                      os.path.join(fullpath2output, 'segm_%s.nii.gz' % (nsim)))