

import os
import argparse
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic images.')

    # Adding command-line arguments
    parser.add_argument('--nsim', type=int, default=4, help='Number of synthetic images created per subject')
    parser.add_argument('--output_shape', type=int, default=300, help='Matrix size of synthetic images created')
    parser.add_argument('--path2dataset', type=str, default='/home/marcantf/Data/training/HCP_20/100408', help='Path to segmentations (inputs to generative model)')
    parser.add_argument('--o', type=str, default='/home/marcantf/Data/training/synthetic/test', help='Path where all synthetic images will be saved (outputs)')
    #parser.add_argument('--one_folder', type=int, default=1, help='Flag indicating if all subjects are in one folder')
    parser.add_argument('--hcp', type=int, default=1, help='Flag indicating if the dataset is from the HCP')

    # Generation parameters
    parser.add_argument('--n_channels', type=int, default=1, help='Number of channels')
    parser.add_argument('--n_neutral_labels', type=int, default=6, help='Number of non-sided labels in segmentations')
    parser.add_argument('--prior_distributions', type=str, default='uniform', help='GMM sampling method (uniform or normal)')
    parser.add_argument('--flipping', type=bool, default=True, help='Flag for applying flipping')
    parser.add_argument('--scaling_bounds', type=float, default=.15, help='Scaling bounds for spatial deformation')
    parser.add_argument('--rotation_bounds', type=float, default=15, help='Rotation bounds for spatial deformation')
    parser.add_argument('--shearing_bounds', type=float, default=.012, help='Shearing bounds for spatial deformation')
    parser.add_argument('--translation_bounds', type=bool, default=False, help='Translation bounds for spatial deformation')
    parser.add_argument('--nonlin_std', type=float, default=3., help='Standard deviation for non-linear deformation')
    parser.add_argument('--bias_field_std', type=float, default=.7, help='Standard deviation for bias field')
    parser.add_argument('--randomise_res', type=bool, default=False, help='Flag for randomising acquisition resolution')
    parser.add_argument('--target_res', type=float, nargs='+', default=None, help='Target resolution of the outputs')

    # Parse the arguments
    args = parser.parse_args()

    # Use parsed arguments in your script
    nsim = args.nsim
    output_shape = args.output_shape
    path2dataset = args.path2dataset
    path2output = args.o
    one_folder = args.one_folder
    hcp = args.hcp
    n_channels = args.n_channels
    n_neutral_labels = args.n_neutral_labels
    prior_distributions = args.prior_distributions
    flipping = args.flipping
    scaling_bounds = args.scaling_bounds
    rotation_bounds = args.rotation_bounds
    shearing_bounds = args.shearing_bounds
    translation_bounds = args.translation_bounds
    nonlin_std = args.nonlin_std
    bias_field_std = args.bias_field_std
    randomise_res = args.randomise_res
    target_res = args.target_res

    ### Paths to the generative model setup [hard-coded paths that are user-dependent]
    path_generation_labels = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors/generation_labels_maf.npy'  # generation labels
    path_segmentation_labels = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors/segmentation_labels_maf.npy'  # segmentation labels
    path_generation_classes = '/home/marcantf/Code/SynthSeg/data/labels_classes_priors/homemade-priors/generation_classes_maf.npy'  # classes generation

    # If the dataset I'm using to create synthetic images is the Humman Connectome Project (HCP)
    if hcp:
        subsubdir = 'T1w'
        aseg_name = 'aseg_noCC.nii.gz'  # !!! this naming convention is subject to modification !!!
        # fullpath2aseg = [os.path.join(f, subsubdir, aseg_name) for f in path2sub]
        fullpath2aseg = [os.path.join(path2dataset, subsubdir, aseg_name)]

    else: # to be coded
        exit("Code not written yet. Sorry, come back later h8ter.")

    ### Preparing the output folder
    path2imgs = os.path.join(path2output, 'imagesTr') # following nnUNet naming convention
    path2segs = os.path.join(path2output, 'labelsTr') # following nnUNet naming convention

    # Create the output folder if not already existing
    if not os.path.exists(path2output):
        os.makedirs(path2output)

    os.makedirs(path2imgs, exist_ok=True)
    os.makedirs(path2segs, exist_ok=True)


    ### Generate synthetic images for all subjects in the dataset
    for idx, sub in enumerate(fullpath2aseg): #loop through all subjects

        print('Processing Subject %s!' % (sub))

        for n in range(nsim): #loop through the number of synthetic image you want to create from each subject

            print('Creating synthetic image #%s for subject #%s!' % (n, sub))

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
            # MAF Q: Do we really need to do this?
            im.astype(np.float16)
            lab.astype(np.int16)

            syn_filename = 'image_%s_0000.nii.gz' %(sub) #synthetic image filename (following nnUNet convention)
            seg_filename = 'image_%s.nii.gz' %(sub) # modified/augmented label map filename (following nnUNet convention)

            # save output synthetic image and augmented label map
            utils.save_volume(im, brain_generator.aff, brain_generator.header, os.path.join(path2imgs, syn_filename))
            utils.save_volume(lab, brain_generator.aff, brain_generator.header, os.path.join(path2segs, seg_filename))



if __name__ == '__main__':
    main()
