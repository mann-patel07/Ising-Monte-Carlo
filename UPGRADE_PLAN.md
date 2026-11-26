# Plan for the seed replicates

Date: 2026-11-26

Written before these results existed. The pre-registered runs and numbers are unchanged.

## Question
1. Is the (8, 16) Binder crossing's miss of the exact Tc noise or a real finite-size effect?
2. Does the bootstrap standard error of a crossing match the spread over independent runs?

## Method
- Rerun the Metropolis fine grid (T = 2.20 to 2.34 in steps of 0.01, 15 temperatures) for L = 8, 16, 32 and 64 with 3 new independent seeds (replicates 1, 2 and 3). Same sweeps, thermalization and blocks as the first version. Replicate 0 is the first version's run.
- For each replicate and each pair (8, 16), (16, 32) and (32, 64): the Binder crossing, with the same rule as the first version.
- For each pair: the mean of the 4 crossings, their sample standard deviation, and the standard error of the mean (SD / 2). Also the crossing from the pooled data (block means averaged over the 4 replicates), with standard error SD / 2.
- Reading for question 1: if the pooled (8, 16) crossing differs from the exact Tc by more than 3 standard errors of the mean, and all 4 replicates are on the same side, I call it a real finite-size effect. Otherwise I call it consistent with noise.
- Reading for question 2: the ratio of the first version's bootstrap standard error to the sample SD of the 4 replicates. With 4 replicates this ratio is itself uncertain by roughly a factor of 1.5 either way, so I report it as a rough check only.
- I report the results whatever they are.
