import ants
import argparse
import os
import glob
import numpy as np
import pandas as pd


def generate_output(args):
    '''
    Generates landmarks, detJ, deformation fields (optional), and followup_registered_to_baseline images (optional) for challenge submission
    '''
    print("generate_output called")

    input_path = os.path.abspath(args["input"])
    output_path = os.path.abspath(args["output"])

    print(f"* Found following data in the input path {input_path}=", os.listdir(input_path)) # Found following data in the input path /input= ['BraTSReg_001', 'BraTSReg_002']
    print("* Output will be written to=", output_path) # Output will be written to= /output

    # Now we iterate through each subject folder under input_path
    for subj_path in glob.glob(os.path.join(input_path, "BraTSReg*")):
        subj = os.path.basename(subj_path)
        print(f"Now performing registration on {subj}") # Now performing registration on BraTSReg_001

        baseline_image_path = glob.glob(os.path.join(subj_path, f"{subj}_00_*_t1ce.nii.gz"))[0]
        followup_image_path = glob.glob(os.path.join(subj_path, f"{subj}_01_*_t1ce.nii.gz"))[0]

        print(f"Baseline (fixed) image: {baseline_image_path}")
        print(f"Followup (moving) image: {followup_image_path}")

        baseline_image = ants.image_read(baseline_image_path)
        followup_image = ants.image_read(followup_image_path)

        followup_landmark_file = glob.glob(os.path.join(subj_path, f"{subj}_01_*_landmarks.csv"))[0]

        print(f"Warping landmarks from {followup_landmark_file}")

        warped_followup_landmark_file = os.path.join(args["output"], f"{subj}.csv")

        reg = ants.registration(baseline_image, followup_image, type_of_transform="antsRegistrationSyN[s,2]", verbose=1)

        ## 1. Warp landmarks

        moving_indices = pd.read_csv(followup_landmark_file).drop('Landmark', axis=1)
        moving_indices = moving_indices.rename(columns={'X' : 'x', 'Y' : 'y', 'Z' : 'z'})
        moving_indices['y'] = moving_indices['y'] + 239

        moving_points = np.zeros(moving_indices.shape)
        for j in range(moving_indices.shape[0]):
            moving_points[j,:] = ants.transform_index_to_physical_point(baseline_image, (moving_indices.iloc[j].values).astype(int))

        moving_points_df = pd.DataFrame(data = {'x': moving_points[:,0], 'y': moving_points[:,1], 'z': moving_points[:,2]})
        moving_warped_points = ants.apply_transforms_to_points(3, moving_points_df, reg['invtransforms'], whichtoinvert=(True, False))
        moving_warped_points = moving_warped_points.to_numpy()
        moving_warped_points_df = pd.DataFrame(data=moving_warped_points, columns=['X', 'Y', 'Z'])
        moving_warped_points_df.insert(0, "Landmark", list(range(1, moving_points.shape[0]+1)))
        moving_warped_points_df.to_csv(warped_followup_landmark_file, index=False)

        ## 2. calculate the determinant of jacobian of the deformation field
        output_detj = ants.create_jacobian_determinant_image(baseline_image, reg['fwdtransforms'][0], do_log=True)

        ## write your output_detj to the output folder
        ants.image_write(output_detj, os.path.join(args["output"], f"{subj}_detj.nii.gz"))

        if args["def"]:
            # write both the forward and backward deformation fields to the output/ folder
            print("--def flag is set to True")
            fwd_composite_warp_file = os.path.join(args["output"], f"{subj}_df_f2b.nii.gz")
            print("fWriting composite forward warps to {fwd_composite_warp_file}")
            tmp_fwd_composite_warp_file = ants.apply_transforms(baseline_image, followup_image, transformlist=reg['fwdtransforms'],
                                          whichtoinvert=(False, False), compose=fwd_composite_warp_file)
            os.rename(tmp_fwd_composite_warp_file,fwd_composite_warp_file)
            print("fWriting composite inverse warps to {inv_composite_warp_file}")
            inv_composite_warp_file = os.path.join(args["output"], f"{subj}_df_b2f.nii.gz")
            tmp_inv_composite_warp_file = ants.apply_transforms(baseline_image, followup_image, transformlist=reg['invtransforms'],
                                  whichtoinvert=(True, False), compose=inv_composite_warp_file)
            os.rename(tmp_inv_composite_warp_file,inv_composite_warp_file)

        if args["reg"]:
            # write the followup_registered_to_baseline sequences (all 4 sequences provided) to the output/ folder
            print("--reg flag is set to True")
            for contrast in ['t1ce', 't1', 't2', 'flair']:
                contrast_moving_file = glob.glob(os.path.join(subj_path, f"{subj}_01_*_{contrast}.nii.gz"))[0]
                contrast_moving = ants.image_read(contrast_moving_file)
                warped = ants.apply_transforms(baseline_image, contrast_moving,
                                               transformlist=reg['fwdtransforms'], whichtoinvert=(False, False))
                ants.image_write(warped, os.path.join(args["output"], f"{subj}_{contrast}_f2b.nii.gz"))


def apply_deformation(args):
    '''
    Applies a deformation field on an input image and saves/returns the output
    '''
    print("apply_deformation called")

    # Read the input image
    input = ants.image_read(args["image"])

    # apply field on image and get output
    output = ants.apply_transforms(input, input, transformlist=[args["field"]], interpolator=args["interpolation"])

    # If a save_path is provided then write the output there, otherwise return the output
    if "path_to_output_nifti" in args:
      ants.image_write(output, args["path_to_output_nifti"])
    else:
      return output


if __name__ == "__main__":
    # Parse the input arguments

    parser = argparse.ArgumentParser(description='Argument parser for BraTS_Reg challenge')

    subparsers = parser.add_subparsers()

    command1_parser = subparsers.add_parser('generate_output')
    command1_parser.set_defaults(func=generate_output)
    command1_parser.add_argument('-i', '--input', type=str, default="/input", help='Provide full path to directory that contains input data')
    command1_parser.add_argument('-o', '--output', type=str, default="/output", help='Provide full path to directory where output will be written')
    command1_parser.add_argument('-d', '--def', action='store_true', help='Output forward and backward deformation fields')
    command1_parser.add_argument('-r', '--reg', action='store_true', help='Output followup scans registered to baseline')

    command2_parser = subparsers.add_parser('apply_deformation')
    command2_parser.set_defaults(func=apply_deformation)
    command2_parser.add_argument('-f', '--field', type=str, required=True, help='Provide full path to deformation field')
    command2_parser.add_argument('-i', '--image', type=str, required=True, help='Provide full path to image on which field will be applied')
    command2_parser.add_argument('-t', '--interpolation', type=str, required=True, help='Should be "genericLabel" (for segmentation mask type images) ' +
                                 'or "nearestNeighbor" (faster label interpolation) or linear (for grayscale scans)')
    command2_parser.add_argument('-p', '--path_to_output_nifti', type=str, default = None, help='Format: /path/to/output_image_after_applying_deformation_field.nii.gz')


    args = vars(parser.parse_args())

    print("* Received the following arguments =", args)

    args["func"](args)