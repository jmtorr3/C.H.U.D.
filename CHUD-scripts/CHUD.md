# C.H.U.D - Catastrophic Harmful Update Detection

![chud](./chud.png)

## Setup
Create a new conda environment using the provided [`environment.yaml`](./environment.yaml) in this directory. Download the AdvBench dataset as per the original LoX instructions, apply the LoX defense to the Llama-2-7b model, and save the model to disk.

Then, check out the jupyter notebooks in this directory to get started:

1. [`sequential_attack.ipynb`](./sequential_attack.ipynb): This notebook implements our iterative fine-tuning attack using Unsloth. Specify the number of samples of GSM8k and BeaverTails to train with, and the script will produce a finetuned model. 

2. [`judge_asr.ipynb`](./judge_asr.ipynb): This notebook implements the ASR evaluation that we used in our experiments. It uses Meta's Llama-Guard-3-8b model as a judge to evaluate a model's responses to 100 samples of the AdvBench dataset.

3. [`judge_baseline.ipynb`](./judge_baseline.ipynb): This notebook tests the judge model used in our ASR evaluations against both the AdvBench dataset (both against the provided harmful responses and with synthetic safe responses) and against BeaverTails, which contains safety annotations.

4. [`chud-results.ipynb`](./chud_results.ipynb): This notebook is used to generate the markdown tables and graphs for our results (see below). Run after all data has been collected with the above scripts.

Additional:

- [`output/`](./output/): Stores `*.csv` log files from the various ASR and baseline experiments. Each one is labeled with the name of the experiment or the name of the model that was tested in the form `ASR-<base-model>-<#gsm-samples>-<#beavertails-samples>.csv`.

- [`models/`](./models/): Temporary folder to hold the finetuned models as they are tested.  

---

## Research Questions & Results:

> See [`chud_results.ipynb`](./chud_results.ipynb) for details and instructions to reproduce the graphs below. 

C.H.U.D investigates three main research questions:

1. **`LoX` vs. `Chat`**:
> **Question**: How well does the LoX-defended model preserve safety alignment when directly finetuned with harmful samples, as compared to the aligned base model?
![Fig 1. `LoX` vs. `Chat`](./output/LoX-vs-Chat-ASR.png)

2. **Benign Finetuning vs. Harmful Finetuning vs. Iterative Attack**
> **Question**: How well does the LoX-defended model preserve safety alignment when we induce catastrophic forgetting by finetuning with benign samples, followed by harmful samples?

![Fig 2. `Chat` w/ iterative attack vs. direct harmful attack](./output/Chat-CatastrophicForgetting.png)

![Fig 2. `LoX` w/ iterative attack vs. direct harmful attack](./output/LoX-CatastrophicForgetting.png)

3. **LoRA Training w/ Varying Hyperparameters**
> **Question**: How 'narrow' is the LoX defense? How does it defend against finetuning at different learning rates? 
![Fig 3. `LoX` fine-tuned with 50 BT samples at varying learning rates](./output/LoX-ASR-vs-LR.png)

---

## Data

> See [`chud_results.ipynb`](./chud_results.ipynb) for details and instructions to reproduce the data. 
> See linked `.csv` files in each row for specific data and test results.

### Judge (`Llama-Guard-3-8b`) Baseline Evaluation
|       Test Set     | Samples | False Negative Rate | False Positive Rate | Overall Accuracy |                                   Output Data                                      |
| -----------------: | ------: | ------------------: | ------------------: | ---------------: | :--------------------------------------------------------------------------------- |
| AdvBench           |   100   |        2.00 %       |        0.00 %       |       99.00 %    | [`output/judge_baseline_advbench.csv`](./output/judge_baseline_advbench.csv)       |
| BeaverTails **\*** |  1000   |       47.60 %       |        4.00 %       |       74.20 %    | [`output/judge_baseline_beavertails.csv`](./output/judge_baseline_beavertails.csv) |

> **\*** BeaverTails samples are 50% safe, 50% unsafe.

### ASR Evaluations on Finetuned Chat Models (Undefended):
| Base Model | GSM Samples | Beavertails Samples | ASR  |                            Output Data                             |
| ---------: | ----------: | ------------------: | ---: | :----------------------------------------------------------------- |
|    Chat    |       0     |           0         |   1% | [`ASR-Chat-0-0.csv`](./output/ASR-Chat-0-0.csv)             |
|    Chat    |       0     |          10         |   5% | [`ASR-Chat-0-10.csv`](./output/ASR-Chat-0-10.csv)           |
|    Chat    |       0     |          25         |  13% | [`ASR-Chat-0-25.csv`](./output/ASR-Chat-0-25.csv)           |
|    Chat    |       0     |         100         |  91% | [`ASR-Chat-0-100.csv`](./output/ASR-Chat-0-100.csv)         |
|    Chat    |       0     |        1000         |  93% | [`ASR-Chat-0-1000.csv`](./output/ASR-Chat-0-1000.csv)       |
|    Chat    |      10     |           0         |   0% | [`ASR-Chat-10-0.csv`](./output/ASR-Chat-10-0.csv)           |
|    Chat    |     100     |           0         |   0% | [`ASR-Chat-100-0.csv`](./output/ASR-Chat-100-0.csv)         |
|    Chat    |    1000     |           0         |   0% | [`ASR-Chat-1000-0.csv`](./output/ASR-Chat-1000-0.csv)       |
|    Chat    |    1000     |          10         |   6% | [`ASR-Chat-1000-10.csv`](./output/ASR-Chat-1000-10.csv)     |
|    Chat    |    1000     |          25         |  11% | [`ASR-Chat-1000-25.csv`](./output/ASR-Chat-1000-25.csv)     |
|    Chat    |    1000     |          50         |  70% | [`ASR-Chat-1000-50.csv`](./output/ASR-Chat-1000-50.csv)     |
|    Chat    |    1000     |         100         |  96% | [`ASR-Chat-1000-100.csv`](./output/ASR-Chat-1000-100.csv)   |
|    Chat    |    1000     |        1000         | 100% | [`ASR-Chat-1000-1000.csv`](./output/ASR-Chat-1000-1000.csv) |

> Learning rate: **5e-4**. See [`sequential_attack.ipynb`](./sequential_attack.ipynb) for additional hyperparameters used.

### ASR Evaluations on Finetuned LoX Models (Defended):
| Base Model | GSM Samples | Beavertails Samples | ASR |                            Output Data                             |
| ---------: | ----------: | ------------------: | --: | :----------------------------------------------------------------- |       
|    LoX     |       0     |           0         |  5% | [`ASR-LoX-0-0.csv`](./output/ASR-LoX-0-0.csv)               |
|    LoX     |       0     |          10         |  3% | [`ASR-LoX-0-10.csv`](./output/ASR-LoX-0-10.csv)             |
|    LoX     |       0     |          25         |  6% | [`ASR-LoX-0-25.csv`](./output/ASR-LoX-0-25.csv)             |
|    LoX     |       0     |         100         | 37% | [`ASR-LoX-0-100.csv`](./output/ASR-LoX-0-100.csv)           |
|    LoX     |       0     |        1000         | 69% | [`ASR-LoX-0-1000.csv`](./output/ASR-LoX-0-1000.csv)         |
|    LoX     |      10     |           0         |  9% | [`ASR-LoX-10-0.csv`](./output/ASR-LoX-10-0.csv)             |
|    LoX     |     100     |           0         |  9% | [`ASR-LoX-100-0.csv`](./output/ASR-LoX-100-0.csv)           |
|    LoX     |    1000     |           0         |  6% | [`ASR-LoX-1000-0.csv`](./output/ASR-LoX-1000-0.csv)         |
|    LoX     |    1000     |          10         |  2% | [`ASR-LoX-1000-10.csv`](./output/ASR-LoX-1000-10.csv)       |
|    LoX     |    1000     |          25         |  6% | [`ASR-LoX-1000-25.csv`](./output/ASR-LoX-1000-25.csv)       |
|    LoX     |    1000     |          50         | 25% | [`ASR-LoX-1000-50.csv`](./output/ASR-LoX-1000-50.csv)       |
|    LoX     |    1000     |         100         | 63% | [`ASR-LoX-1000-100.csv`](./output/ASR-LoX-1000-100.csv)     |
|    LoX     |    1000     |        1000         | 89% | [`ASR-LoX-1000-1000.csv`](./output/ASR-LoX-1000-1000.csv)   |

> Learning rate: **5e-4**. See [`sequential_attack.ipynb`](./sequential_attack.ipynb) for additional hyperparameters used.

### ASR Evaluations on Finetuned LoX Models at Various Learning Rates:
| Learning Rate | ASR |                            Output Data                             |
| :-----------: | --: | :----------------------------------------------------------------- |
|      1e-5     |  7% | [`ASR-LoX-0-50-1e-5.csv`](./output/ASR-LoX-0-50-1e-5.csv)   | 
|      2e-5     | 14% | [`ASR-LoX-0-50-2e-5.csv`](./output/ASR-LoX-0-50-2e-5.csv)   | 
|      5e-5     |  6% | [`ASR-LoX-0-50-5e-5.csv`](./output/ASR-LoX-0-50-5e-5.csv)   | 
|      1e-4     |  4% | [`ASR-LoX-0-50-1e-4.csv`](./output/ASR-LoX-0-50-1e-4.csv)   |
|      2e-4     | 10% | [`ASR-LoX-0-50-2e-4.csv`](./output/ASR-LoX-0-50-2e-4.csv)   | 
|      3e-4     | 12% | [`ASR-LoX-0-50-3e-4.csv`](./output/ASR-LoX-0-50-3e-4.csv)   |
|      5e-4     | 38% | [`ASR-LoX-0-50-5e-4.csv`](./output/ASR-LoX-0-50-5e-4.csv)   | 

> Finetuned w/ 50 harmful samples from Beavertails.

---


## Notes

The LoRA finetuning attack implemented in [`sequential_attack.ipynb`](./sequential_attack.ipynb) builds off traditional LoRA by unfreezing the parameters in the Language Modeling layer (`lm_head`). This is the final layer that projects the final logits linearly, and then applies softmax to choose the next token. This method trains ~6.39% of the total parameter set by including this layer in finetuning. Without this modification, only ~4.53% of parameters are trainable.