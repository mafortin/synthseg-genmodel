"""
Own reimplementation of the BrainGenerator call from SynthSeg to be more user-friendly for my usage.

Author: MAF, August 2024. marc.a.fortin@ntnu.no
"""

### Imports
import os
import argparse
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic images.')

    # Adding command-line arguments
    parser.add_argument('--i', type=str, required=True, help='Path to segmentations (inputs to generative model). The script assumes that the folder contains all label maps in the first level (and not all in different folders).')
    parser.add_argument('--o', type=str, required=True, help='Path where all synthetic images will be saved (outputs). Will be created if doesn t exist')
    parser.add_argument('--nsim', type=int, default=1, help='Number of synthetic images created per subject')
    parser.add_argument('--output_shape', type=int, default=288, help='Matrix size of synthetic images created. Default value is 1.5 times 192, which is the default 3D patch size of the nnUNet.')
    parser.add_argument('--dynamic_output_shape', action='store_true', help='Flag indicating if all subjects are in one folder and output_shape will be decided based on the dataset id. This will overwrite the flag --output_shape.')
    parser.add_argument('--hcp', action='store_true', help='Indicate if the dataset is from the Human Connectome Project.')
    parser.add_argument('--keep_id', action='store_true', help='To keep th subject ID from its original study or just use a linearly increasing new ID.')
    parser.add_argument('--only_subdataset', default='XX', help='Indicate to process only one of the subdataset. Give the letter identifying this subdataset (e.g., A).')
    # Generation parameters
    parser.add_argument('--n_channels', type=int, default=1, help='Number of channels')
    parser.add_argument('--n_neutral_labels', type=int, default=6, help='Number of non-sided labels in label maps')
    parser.add_argument('--prior_dist', type=str, default='uniform', help='GMM sampling method (uniform or normal)')
    parser.add_argument('--flipping', type=bool, default=True, help='Flag for applying flipping')
    parser.add_argument('--scaling_bounds', type=float, default=.15, help='Scaling bounds for spatial deformation')
    parser.add_argument('--rotation_bounds', type=float, default=15, help='Rotation bounds for spatial deformation')
    parser.add_argument('--shearing_bounds', type=float, default=.012, help='Shearing bounds for spatial deformation')
    parser.add_argument('--translation_bounds', type=bool, default=False, help='Translation bounds for spatial deformation')
    parser.add_argument('--nonlin_std', type=float, default=3., help='Standard deviation for non-linear deformation')
    parser.add_argument('--bias_field_std', type=float, default=.7, help='Standard deviation for bias field')
    parser.add_argument('--randomise_res', type=bool, default=False, help='Flag for randomising acquisition resolution')
    parser.add_argument('--target_res', type=float, nargs='+', default=None, help='Target resolution of the outputs')
    parser.add_argument('--data_res', type=float, nargs='+', default=None, help='Data resolution of the outputs')
    parser.add_argument('--max_res_iso', type=float, nargs='+', default=None, help='Max isotropic resolution of the outputs')

    # Parse the arguments
    args = parser.parse_args()

    # Use parsed arguments in your script
    nsim = args.nsim
    output_shape = args.output_shape
    path2input = args.i
    path2output = args.o
    dynamic_output_shape = args.dynamic_output_shape
    hcp = args.hcp
    keep_id = args.keep_id
    only_subdataset = args.only_subdataset

    n_channels = args.n_channels
    n_neutral_labels = args.n_neutral_labels
    prior_distributions = args.prior_dist
    flipping = args.flipping
    scaling_bounds = args.scaling_bounds
    rotation_bounds = args.rotation_bounds
    shearing_bounds = args.shearing_bounds
    translation_bounds = args.translation_bounds
    nonlin_std = args.nonlin_std
    bias_field_std = args.bias_field_std
    randomise_res = args.randomise_res
    target_res = args.target_res
    data_res = args.data_res
    max_res_iso = args.max_res_iso

    #### USER-DEPENDENT VARIABLES

    # Dictionary of all output shapes (i.e., matrix sizes of synthetic images) you want based on each dataset.
    # ***************To be updated if the dataset ID changes.***************
    dataset_output_shapes = {
        'A': 320, # UltraCortex (0.6mm) iso
        'B': 500, # MPI-CBS (0.4mm) iso
        'C': 288, # HCP (0.7mm) iso
        'D': 288, # UltraCortex (0.8mm) iso
        'E': 256, # ABIDEII (0.9mm) iso
        'F': 384, # SCAIFIELD (0.6mm) iso
        'G': 256, # ABIDEII (0.9mm) iso
    }

    ### Paths to the generative model setup [hard-coded paths that are user-dependent]
    # ***************To be updated for different priors/paths.***************
    path_generation_labels = '/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/generation_labels_maf_guhfi2.npy'  # generation labels
    path_segmentation_labels = '/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/segmentation_labels_maf_guhfi2.npy'  # segmentation labels
    path_generation_classes = '/home/marcantf/Code/PhD-python/synthseg-genmodel/labels_classes_priors/homemade-priors/generation_classes_maf_guhfi2.npy'  # classes generation
    ####

    if dynamic_output_shape:
        print("Dynamic output shape enabled. Different matrix sizes will be used for different datasets as specified with the user-defined dictionary at the beginning of this script.")

    # If the dataset I'm using to create synthetic images is the Human Connectome Project (HCP)
    if hcp:
        subsubdir = 'T1w'
        aseg_name = 'aseg_noCC.nii.gz'  # !!! this naming convention is subject to modification !!!
        # fullpath2aseg = [os.path.join(f, subsubdir, aseg_name) for f in path2sub]
        fullpath2aseg = [os.path.join(path2input, subsubdir, aseg_name)]

    else:

        all_labelmaps = [n for n in os.listdir(path2input) if '.nii.gz' in n]
        print("%s different label maps detected in directory %s" % (len(all_labelmaps), path2input))
        all_paths2labelmaps = [os.path.join(path2input, nn) for nn in all_labelmaps]

        if keep_id:
            all_subids = [s[s.find('aseg_') + len('aseg_'):s.find('.nii.gz')].strip() for s in all_paths2labelmaps if 'aseg_' in s and '.nii.gz' in s]

    ### Preparing the output folder
    path2imgs = os.path.join(path2output, 'imagesTr') # following nnUNet naming convention
    path2segs = os.path.join(path2output, 'labelsTr') # following nnUNet naming convention

    # Create the output folder if not already existing
    if not os.path.exists(path2output):
        os.makedirs(path2output)

    os.makedirs(path2imgs, exist_ok=True)
    os.makedirs(path2segs, exist_ok=True)


    ### Generate synthetic images for all subjects in the dataset
    for idx, sub in enumerate(all_paths2labelmaps): #loop through all subjects

        if dynamic_output_shape:
            # Get dataset id
            subname = os.path.basename(sub).replace('.nii.gz', '').replace('aseg_', '')
            dataset_id = subname[0]
            print("Dataset %s detected." %(dataset_id))

            # Fetch matrix size of synthetic image to be computed (variable == output_shape)
            output_shape = dataset_output_shapes[dataset_id]
            print("Since dataset %s was detected, an isotropic matrix size of %s will be used for the synthetic image to be computed!" %(dataset_id, str(output_shape)))

        if dataset_id == only_subdataset: # MAF to be modified since it doesnt work as of now and was jsut used for debugging purposes
            print("Subdataset skipped since it is not the subdataset corresponding to the flag '--only_subdataset' by the user.")

        else:
            for n in range(nsim): #loop through the number of synthetic image you want to create from each subject

                if keep_id:
                    print('Currently processing Subject %s!' % (all_subids[idx]))
                    print('Creating synthetic image #%s for Subject #%s.' % (n, all_subids[idx]))
                else:
                    print('Currently processing Subject #%s!' % (idx))
                    print('Creating synthetic image #%s for Subject #%s.' % (n, idx))

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
                                                 randomise_res=randomise_res) # max_res_iso=max_res_iso, can be added


                # generate new image and corresponding labels
                im, lab = brain_generator.generate_brain()

                # Reduce precision level of both the synthetic image and label map (from 32 to 16bit)
                # MAF Q: Do we really need to do this?
                im.astype(np.float16)
                lab.astype(np.int16)

                if nsim==1:
                    if keep_id:
                        syn_filename = 'image_%s_0000.nii.gz' %(all_subids[idx]) #synthetic image filename (following nnUNet convention)
                        seg_filename = 'image_%s.nii.gz' %(all_subids[idx]) # modified/augmented label map filename (following nnUNet convention)
                    else:
                        syn_filename = 'image_%s_0000.nii.gz' %(idx) #synthetic image filename (following nnUNet convention)
                        seg_filename = 'image_%s.nii.gz' %(idx) # modified/augmented label map filename (following nnUNet convention)
                else:
                    if keep_id:
                        syn_filename = 'image_%s_%s_0000.nii.gz' %(all_subids[idx], n) #synthetic image filename (following nnUNet convention)
                        seg_filename = 'image_%s_%s.nii.gz' %(all_subids[idx], n) # modified/augmented label map filename (following nnUNet convention)
                    else:
                        syn_filename = 'image_%s_%s_0000.nii.gz' %(idx, n) #synthetic image filename (following nnUNet convention)
                        seg_filename = 'image_%s_%s.nii.gz' %(idx, n) # modified/augmented label map filename (following nnUNet convention)

                # save output synthetic image and augmented label map
                utils.save_volume(im, brain_generator.aff, brain_generator.header, os.path.join(path2imgs, syn_filename))
                utils.save_volume(lab, brain_generator.aff, brain_generator.header, os.path.join(path2segs, seg_filename))

            if keep_id:
                print('Done processing Subject %s!' % (all_subids[idx]))
            else:
                print('Done processing Subject #%s!' % (idx))


if __name__ == '__main__':
    main()
