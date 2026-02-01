  # Official implementation: "Meta-Learning-Based Handover Management in NextG O-RAN"

This code sample includes the implementation of the algorithm for Conditional-Traditional Handover Management (CONTRA), published in the IEEE Journal on Selected Areas in Communications (JSAC), 2026.
Paper: https://doi.org/10.1109/JSAC.2026.3652037

## Abstract

> While traditional handovers (THOs) have served as a backbone for mobile connectivity, they increasingly suffer from failures and delays, especially in dense deployments and high-frequency bands. To address these limitations, 3GPP introduced Conditional Handovers (CHOs) that enable proactive cell reservations and user-driven execution. However, both handover (HO) types present intricate trade-offs in signaling, resource usage, and reliability. This paper presents unique, countrywide mobility management datasets from a top-tier mobile network operator (MNO) that offer fresh insights into these issues and call for adaptive and robust HO control in next-generation networks. Motivated by these findings, we propose CONTRA, a framework that, for the first time, jointly optimizes THOs and CHOs within the O-RAN architecture. We study two variants of CONTRA: one where users are a priori assigned to one of the HO types, reflecting distinct service or user-specific requirements, as well as a more dynamic formulation where the controller decides on-the-fly the HO type, based on system conditions and needs. To this end, it relies on a practical meta-learning algorithm that adapts to runtime observations and guarantees performance comparable to an oracle with perfect future information (universal no-regret). CONTRA is specifically designed for near-real-time deployment as an O-RAN xApp and aligns with the 6G goals of flexible and intelligent control. Extensive evaluations leveraging crowdsourced datasets show that CONTRA improves user throughput and reduces both THO and CHO switching costs, outperforming 3GPP-compliant and Reinforcement Learning (RL) baselines in dynamic and real-world scenarios.

## Usage
1. Download the repository, enter the main directory, and install the project (once)
```sh
   cd ./conditional_traditional_handover_management
   pip install -e .
```
2. Run the following (feel free to give different data as input):
```sh
   python3 ./src/contra_algorithm/experiments/run_experiments.py \
  --sinr_path ./data/sinr_T-1000_I-100_J-5.npy \
  --THOdelay_path ./data/HO_delays_T-1000_I-100_J-5.npy \
  --CHOsignaling_path ./data/HO_delays_T-1000_I-100_J-5.npy \
  --w_path ./data/bandwidth_J-5.npy
```

## Citing Work

To acknowledge the use of the source code, please use the following reference:

### Plain Text

M. Kalntis, G. Iosifidis, J. Suárez-Varela, A. Lutu and F. A. Kuipers, "Meta-Learning-Based Handover Management in NextG O-RAN," in IEEE Journal on Selected Areas in Communications, 2026, doi: 10.1109/JSAC.2026.3652037

### BibTeX

```
@ARTICLE{11340677,
  author={Kalntis, Michail and Iosifidis, George and Suárez-Varela, José and Lutu, Andra and Kuipers, Fernando A.},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Meta-Learning-Based Handover Management in NextG O-RAN}, 
  year={2026},
  volume={},
  number={},
  pages={1-1}
  doi={10.1109/JSAC.2026.3652037}}

```
