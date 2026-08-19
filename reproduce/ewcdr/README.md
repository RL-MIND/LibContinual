# EWC-DR: Elastic Weight Consolidation Done Right for Continual Learning [(CVPR'2026)](https://arxiv.org/abs/2603.18596)

## Abstract

Weight-regularization methods alleviate catastrophic forgetting by estimating
which parameters are important for previous tasks and penalizing their changes.
Elastic Weight Consolidation (EWC) is a classic method in this family, but its
Fisher-based importance estimation can suffer from vanishing gradients when the
model is already confident on old-task samples. EWC-DR addresses this issue with
a simple Logits Reversal operation during importance estimation:

```python
loss = cross_entropy(-logits, labels)
```

This migrated implementation follows the official EWC-DR logic in the
LibContinual training interface. It keeps the expanding classifier, new-class
cross entropy for incremental tasks, clipped importance weights, and
exemplar-free class-incremental learning setting.


## Citation

```bibtex
@inproceedings{liu2026elastic,
  title={Elastic Weight Consolidation Done Right for Continual Learning},
  author={Liu, Xuan and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## How to Reproduce

- **Step 1: Install dependencies**

    Use the repository dependency file:

    ```bash
    conda create -n libcontinual-ewcdr python=3.8 -y
    conda activate libcontinual-ewcdr
    pip install -r requirements.txt
    ```

- **Step 2: Prepare CIFAR-100**

    The provided configs use `binary_cifar100`. CIFAR-100 will be downloaded to:

    ```text
    ./data/binary_cifar100/cifar-100-python
    ```

- **Step 3: Run one setting**

    ```bash
    python run_trainer.py --config ewcdr-resnet18-cifar100-b10-10-10.yaml --device 0
    python run_trainer.py --config ewcdr-resnet18-cifar100-b50-5-11.yaml --device 0
    ```

- **Step 4: Run all CIFAR-100 Table 1 EWC-DR settings**

    ```bash
    python reproduce/ewcdr/run_cifar100_table1.py --device 0
    ```

    To reproduce the paper-style mean over three independent trials:

    ```bash
    python reproduce/ewcdr/run_cifar100_table1.py --device 0 --seeds 1993 1994 1995
    ```

- **Step 5: Run ImageNet-Subset Table 2 or Tiny-ImageNet Table 3**

    Prepare ImageNet-Subset in ImageFolder layout under
    `./data/ImageNet-100/imagenet-100/{train,val}/<class>/`. Tiny-ImageNet is
    downloaded by the existing `tiny-imagenet` data backend under
    `./data/tiny-imagenet-200` (`data_root` itself is `./data`, because the
    backend appends the dataset directory name).

    ```bash
    python reproduce/ewcdr/run_imagenet_subset_table2.py --device 0
    python reproduce/ewcdr/run_tiny_imagenet_table3.py --device 0
    ```

    Use `--seeds 1993 1994 1995` for the paper-style three-run mean, or
    `--dry-run` to inspect all generated commands. The runners execute two jobs
    concurrently per GPU by default. Pass multiple IDs, for example
    `--device 0 1 2 --max-parallel 2`, to run at most two jobs on each GPU
    (six total). They archive stable logs plus `summary.md` under
    `reproduce/ewcdr/logs/<dataset_table>/`.

## Settings

| Paper Setting | LibContinual Config | Class Split |
| :---: | :--- | :--- |
| Big T=5 | `ewcdr-resnet18-cifar100-b50-10-6.yaml` | 50 initial + 5 increments x 10 classes |
| Big T=10 | `ewcdr-resnet18-cifar100-b50-5-11.yaml` | 50 initial + 10 increments x 5 classes |
| Big T=20 | `ewcdr-resnet18-cifar100-b40-3-21.yaml` | 40 initial + 20 increments x 3 classes |
| Eq T=5 | `ewcdr-resnet18-cifar100-b20-20-5.yaml` | 5 tasks x 20 classes |
| Eq T=10 | `ewcdr-resnet18-cifar100-b10-10-10.yaml` | 10 tasks x 10 classes |
| Eq T=20 | `ewcdr-resnet18-cifar100-b5-5-20.yaml` | 20 tasks x 5 classes |

ImageNet-Subset uses the same six 100-class splits as CIFAR-100. Its configs
are named `ewcdr-resnet18-imagenet100-*.yaml`. Tiny-ImageNet uses:

| Paper Setting | LibContinual Config | Class Split |
| :---: | :--- | :--- |
| Big T=5 | `ewcdr-resnet18-tinyimagenet-b100-20-6.yaml` | 100 initial + 5 increments x 20 classes |
| Big T=10 | `ewcdr-resnet18-tinyimagenet-b100-10-11.yaml` | 100 initial + 10 increments x 10 classes |
| Big T=20 | `ewcdr-resnet18-tinyimagenet-b100-5-21.yaml` | 100 initial + 20 increments x 5 classes |
| Eq T=5 | `ewcdr-resnet18-tinyimagenet-b40-40-5.yaml` | 5 tasks x 40 classes |
| Eq T=10 | `ewcdr-resnet18-tinyimagenet-b20-20-10.yaml` | 10 tasks x 20 classes |
| Eq T=20 | `ewcdr-resnet18-tinyimagenet-b10-10-20.yaml` | 20 tasks x 10 classes |

## Results

The reproduced results below are single-run results with `seed=1993`. The paper
reports means over three independent trials.

### CIFAR-100

| Setting | Seed | LibContinual A_last | Paper A_last | Diff | LibContinual A_avg | Paper A_avg | Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Big start / T=5 | 1993 | 50.87 | 50.23 | **+0.64** | 64.48 | 63.75 | **+0.73** |
| Big start / T=10 | 1993 | 46.91 | 44.88 | **+2.03** | 61.09 | 60.94 | **+0.15** |
| Big start / T=20 | 1993 | 36.02 | 35.86 | **+0.16** | 52.80 | 53.45 | **-0.65** |
| Equally split / T=5 | 1993 | 46.45 | 46.89 | **-0.44** | 61.42 | 61.47 | **-0.05** |
| Equally split / T=10 | 1993 | 30.34 | 29.41 | **+0.93** | 47.45 | 46.01 | **+1.44** |
| Equally split / T=20 | 1993 | 19.15 | 18.00 | **+1.15** | 34.44 | 33.52 | **+0.92** |

### Tiny-ImageNet

| Setting | Seed | LibContinual A_last | Paper A_last | Diff | LibContinual A_avg | Paper A_avg | Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Big start / T=5 | 1993 | 39.59 | 38.24 | **+1.35** | 47.47 | 47.00 | **+0.47** |
| Big start / T=10 | 1993 | 35.48 | 31.43 | **+4.05** | 45.13 | 42.88 | **+2.25** |
| Big start / T=20 | 1993 | 30.69 | 23.64 | **+7.05** | 41.32 | 37.56 | **+3.76** |
| Equally split / T=5 | 1993 | 26.46 | 28.67 | **-2.21** | 38.34 | 39.52 | **-1.18** |
| Equally split / T=10 | 1993 | 21.46 | 21.46 | **+0.00** | 34.30 | 32.79 | **+1.51** |
| Equally split / T=20 | 1993 | 13.60 | 12.09 | **+1.51** | 24.50 | 22.62 | **+1.88** |


The LibContinual metrics correspond to the final log fields:

| Paper Metric | LibContinual Log Field |
| :---: | :--- |
| `A_last` | `[Batch] Last Average Acc` |
| `A_avg` | `[Batch] Overall Avg Acc` |

## Notes

- The official big-start `T` counts incremental phases, while LibContinual
  `task_num` includes the initial phase. For example, Big T=10 is configured as
  `task_num=11`.
- `binary_cifar100` must follow the same shuffled class order and label remap as
  the official implementation; this is handled in the migrated data pipeline.
- Initial and incremental stages use different schedulers, matching the official
  setup.
- The Table 1 runner can launch multiple jobs on the same GPU for convenience.
  For strict reporting, prefer three seeds and avoid overloading one GPU.
