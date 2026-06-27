# Balanced Game Design: Section 7 Numerical Experiments

This repository contains the code and data files used for the numerical experiments in Section 7 of the paper 

> X. Guan and X. Lei (2026). Balanced Game Design. Available at SSRN: https://ssrn.com/abstract=7001878.

Section 7 evaluates the proposed approximate optimization method on synthetic instances and demonstrates an end-to-end case study on a real game.


## Software Requirements

The code is implemented in Python. The main dependencies are:

- IBM DOcplex/CPLEX, used for the proposed approximate model and CPLEX callback customizations
- Gurobi and `gurobipy`, used for direct nonconvex baselines and some Gurobi implementations of the approximate model

The proposed approximate method is implemented with DOcplex and solved with CPLEX 12.1.1.0, while several baseline models are implemented with `gurobipy` and solved with Gurobi 12.0.3. 

## Repository Structure

This repository contains the following experiment groups:

- Small-scale synthetic experiments for Section 7.1
- Large-scale synthetic experiments for Section 7.1
- Multi-scenario synthetic experiments for Section 7.1
- Street Fighter case-study files for Section 7.2

Most scripts write summary CSV files, solver logs, and selected design matrices to the current working directory. 

## Section 7.1: Small-Scale Synthetic Experiments

The first set of Section 7.1 experiments uses 10 synthetic instances with `N = 11` characters and `K = 12` binary attributes. The scripts compare eleven methods summarized in Table 1 of the paper.

### Proposed Approximate-Model Variants in CPLEX

The following five scripts correspond to the APX-based methods in Table 1 of the paper. They solve approximate model (9) using CPLEX with a node limit of 500,000 and CPLEX heuristics disabled.

- `small scale experiments APX.py`: Implements `APX`, the plain approximate model in CPLEX without custom callbacks. 

- `small scale experiments APX-RJ.py`: Implements `APX-RJ`, the approximate model with incumbent selection. 

- `small scale experiments APX-RJ-PN.py`: Implements `APX-RJ&PN`, the approximate model with incumbent selection and customized node pruning. 

- `small scale experiments APX-RJ-SL.py`: Implements `APX-RJ&SL`, the approximate model with incumbent selection and customized node selection. 

- `small scale experiments APX-RJ-PN-SL.py`: Implements `APX-RJ&PN&SL`, combining incumbent selection, customized node pruning, and customized node selection. 

### Small-Scale Baseline Methods

These scripts implement the baseline methods compared against the APX-based methods in Table 1.

- `small-scale-experiments-local search method.py`: Implements `LS`, a human-like local search heuristic. Starting from an all-zero design, it flips one character-attribute decision at a time, solves the induced zero-sum-game LP, and searches for designs with smaller maximum equilibrium pick rate.

- `small scale DCT-solve by discretizing pi.py`: Implements `DCT`, the discretization baseline for the bilinear terms involving equilibrium probabilities. It discretizes each `p_i` at granularity `0.01`, linearizes the resulting products, solves the approximate MIP in CPLEX.

- `small scale solved by spatial BB.py`: Implements `SBB`, the spatial branch-and-bound baseline with McCormick relaxations. 

- `small scale directly solve in Gurobi with heuristic.py`: Implements `DT-Gurobi`, the direct Gurobi solve of the original balanced-design formulation with bilinear terms and Gurobi's default heuristics. 

- `small scale directly solve in Gurobi without heuristic.py`: Implements `DT-Gurobi-w/o`, the direct Gurobi solve with heuristics disabled. 

- `small scale solve APX-RJ in Gurobi.py`: Implements `APX-RJ-Gurobi`, a Gurobi version of the approximate model with incumbent selection. 

## Section 7.1: Large-Scale Synthetic Experiments

The second set of Section 7.1 experiments uses large synthetic instances with `N = 29` characters and `K = 29` attributes. This scale is intended to be comparable to modern fighting games. The paper compares the proposed approximate method against spatial branch-and-bound and direct Gurobi.

- `large-scale-experiments-APX-RJ-PN-with-solution-polishing.py`: Implements the large-scale version of the proposed approximate method. It uses incumbent selection and customized node pruning, and it adds the solution-polishing procedure described in Section 6.1 and Appendix D.1. The large-scale experiment enables dynamic search and multithreading, so the customized node-selection callback is not used.

- `large scale solved by spatial BB.py`: Implements the large-scale `SBB` baseline. It applies the same spatial branch-and-bound idea as the small-scale SBB script, but on `N = K = 29` instances.

- `large scale directly solve balanced design problem in Gurobi.py`: Implements the large-scale `DT-Gurobi` baseline. It directly solves the nonconvex balanced-design model in Gurobi with SOS2 approximations for the logistic win-rate terms and `NonConvex = 2`.

## Section 7.1: Multi-Scenario Synthetic Experiments

The third set of Section 7.1 experiments uses 10 synthetic instances with `N = 11`, `K = 12`, and `S = 100` scenarios. Each scenario is generated by perturbing a common base instance. The paper compares directly solving the multi-scenario approximate formulation with a scenario decomposition approach. In the paper, the experiments are conducted on an Alibaba Cloud Desktop environment using a single Enterprise Office configuration with 8 virtual CPUs and 16 GiB memory.

- `multi-scenario experiments_dynamicserach.py`: Implements the direct multi-scenario approximate method. 

- `scenario decomposition experiments_dynamicserach.py`: Implements the scenario decomposition method from Section 6.3. 

## Section 7.2: Street Fighter Case Study

Section 7.2 studies a fan-made version of Street Fighter with seven modifiable characters: Alex, Charlie Nash, Chun-Li, Dan, Guile, M.Bison, and Ryu. The case study estimates logistic win-rate models from simulated matchup data, and uses proposed methods to produce candidate game modifications.

- `StreetFighter case study legacy callback_polishing after solving.py`: Implements the end-to-end case-study optimization pipeline. 

- `MNL_models_parameters_streetfighter.csv`: Input parameter file for the Street Fighter case study. Each row gives an attacker/defender pair and the estimated MNL/logistic coefficients used to construct pairwise win-rate predictions.


## Output Files

The scripts generally produce three kinds of outputs:

- Summary CSV files with one row per random seed or instance.
- Text log files containing CPLEX or Gurobi solver output.
- CSV files containing candidate design matrices.

