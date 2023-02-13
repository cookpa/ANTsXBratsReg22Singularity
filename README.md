# ANTsXBratsReg22Singularity

Containerized implementation of the ANTsX
[BrATS-Reg22](https://github.com/ntustison/BraTS-Reg22) entry.

See

https://github.com/satrajitgithub/BraTS_Reg_submission_instructions/blob/master/BraTS_Reg_instructions_singularity.md

for details of the run script.

The container defaults to using 8 threads in the registration; this may be
changed to `N` threads with `--env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=N`.


## ANTs methodology

The registrations are performed using the algorithm here

https://github.com/ntustison/BraTS-Reg22#apply-antsregistrationsyns2-with-t1-contrast-enhanced-to-validation-data

using the t1ce images as input.


## Reproducible results

There is run-to-run variability as a result of random sampling, and precision errors in
the ITK Mattes Mutual Information filter.

Random sampling can be disabled with a fixed random seed `X` by setting the evironment
variable `ANTS_RANDOM_SEED=X`. To make the registration fully deterministic, set the
random seed and also `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`.


## Container image

The container image is available at 

https://cloud.sylabs.io/library/cookpa/brats_reg/brats_reg_antsx
