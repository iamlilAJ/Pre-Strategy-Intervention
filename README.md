# A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning

This repository contains the official JAX implementation for the NeurIPS 2025 paper: *A Principle of Targeted Intervention in Multi-Agent Reinforcement Learning*.



## Installation


This project requires Python 3.10 and we recommend `conda` for environment management. The installation is a **two-stage process**: first, you install the hardware-specific JAX library, and second, you install this project's dependencies.

1.  **Clone the repository:**
    ```shell
    git clone [https://github.com/iamlilAJ/Pre-Strategy-Intervention.git](https://github.com/iamlilAJ/Pre-Strategy-Intervention.git)
    cd Pre-Strategy-Intervention
    ```

2.  **Create and activate the conda environment:**
    ```shell
    conda create -n intervention python=3.10
    conda activate intervention
    ```

3.  **Install JAX for your specific hardware:**

    * **For NVIDIA GPU Users (Recommended):**
        This command installs the exact versions of JAX, a CUDA-enabled jaxlib, and cuDNN that are compatible with this project.
        ```shell
        pip install jax==0.4.25 jaxlib==0.4.25 nvidia-cudnn-cu12==8.9.2.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

    * **For CPU-Only Users:**
        If you do not have an NVIDIA GPU, install the CPU-only version of JAX.
        ```shell
        pip install jax==0.4.25 jaxlib==0.4.25
        ```

4.  **Install the project and dependencies:**
    Now that JAX is correctly installed, you can install the rest of the project's dependencies. This command uses the `[algs]` extra to include packages like `optax` and `wandb`.
    ```shell
    pip install -e .[algs]
    ```



### Hanabi

To run the IPPO
```shell
python baselines/IPPO/ippo_pre.py +alg=ippo
```

To change the scenarios
```shell
python baselines/IPPO/ippo_pre.py +alg=ippo alg.ENV_KWARGS.convention_type=the_chop
```

To run the MAPPO
```shell
python baselines/MAPPO/mappo_pre.py +alg=mappo
```

PQN-VDN
```shell
python baselines/QLearning/pqn_vdn_pre.py +alg=pqn
```

PQN-IQL
```shell
python baselines/QLearning/pqn_iql_pre.py +alg=pqn
```

### MPE

IQL
```shell
python baselines/QLearning/iql_pre.py +alg=iql
```
To run the second scenario
```shell
python baselines/QLearning/iql_pre.py +alg=iql_scenario_2
```
VDN
```shell
python baselines/QLearning/vdn_pre.py +alg=vdn
```
QMIX
```shell
python baselines/QLearning/qmix_pre.py +alg=qmix
```

## Visualization of Learned Behavior


                                                                                                                           
|                |                                                       |                                                       |
| :------------- | :---------------------------------------------------: | :---------------------------------------------------: |
| **Our Method** | ![MPE Visualization 1](assets/MPE_visualization_1.gif) | ![MPE Visualization 2](assets/MPE_visualization_2.gif) |
| **Baseline** | ![Baseline 1](assets/MPE_visualization_baseline_1.gif) | ![Baseline 2](assets/MPE_visualization_baseline_2.gif) |

In this visualization, the red agent (our intervened agent) has learned a preference for moving towards the yellow landmark. By learning this simple additional desired outcome, the agent team can achieve effective coordination and successfully solve the task.

## Citation

If you use this code in your research, please consider citing our paper:

```latex
@inproceedings{liu2025targeted,
  title={A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning},
  author={Liu, Anjie and Wang, Jianhong and Kaski, Samuel and Wang, Jun and Yang, Mengyue},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
## License and Acknowledgements
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

Our implementation is built upon the excellent [JaxMARL](https://github.com/FLAIROx/JaxMARL) library. We thank the original authors for their significant contributions to the community.
