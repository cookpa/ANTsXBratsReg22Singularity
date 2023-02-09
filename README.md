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
