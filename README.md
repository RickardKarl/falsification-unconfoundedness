# Code for "Falsification of Unconfoundedness by Testing Independence of Causal Mechanisms"

[Paper link](https://arxiv.org/abs/2502.06231)

### There is now a Python package also

I have also turned this package into an easy-to-use Python package: **[causal-falsify](https://github.com/RickardKarl/causal-falsify)**, please check it out.

## Abstract 

A major challenge in estimating treatment effects in observational studies is the reliance on untestable conditions such as the assumption of no unmeasured confounding. In this work, we propose an algorithm that can falsify the assumption of no unmeasured confounding in a setting with observational data from multiple heterogeneous sources, which we refer to as environments. Our proposed falsification strategy leverages a key observation that unmeasured confounding can cause observed causal mechanisms to appear dependent. Building on this observation, we develop a novel two-stage procedure that detects these dependencies with high statistical power while controlling false positives. The algorithm does not require access to randomized data and, in contrast to other falsification approaches, functions even under transportability violations when the environment has a direct effect on the outcome of interest. To showcase the practical relevance of our approach, we show that our method is able to efficiently detect confounding on both simulated and real-world data.


## Instructions

- Install requirements using the commando pip install -r requirements.txt
- Our proposed method is found in src/ours/
- Files for reproducing experiments from paper are found in experiments/run.py and experiments/notebooks/



