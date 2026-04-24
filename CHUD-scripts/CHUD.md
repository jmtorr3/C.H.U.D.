# C.H.U.D - Catastrophic Harmful Update Detection

![chud](./chud.png)

## Setup
Create a new conda environment using the provided [`environment.yaml`](./environment.yaml) in this directory. Download the AdvBench dataset as per the original LoX instructions, apply the LoX defense to the Llama-2-7b model, and save the model to disk.

Then, check out the jupyter notebooks in this directory to get started:

1. [`sequential_attack.ipynb`](./sequential_attack.ipynb): This notebook implements our sequential fine-tuning attack using Unsloth. Specify the number of samples of GSM8k and BeaverTails to train with, and the script will produce a finetuned model.

2. [`judge_asr.ipynb`](./judge_asr.ipynb): This notebook implements the ASR evaluation that we used in our experiments. It uses Meta's Llama-Guard-3-8b model as a judge to evaluate a model's responses to 100 samples of the AdvBench dataset.

3. [`judge_baseline.ipynb`](./judge_baseline.ipynb): This notebook tests the judge model used in our ASR evaluations against both the AdvBench dataset (both against the provided harmful responses and with synthetic safe responses) and against BeaverTails, which contains safety annotations.

Additional:
- [`utils.py`](./utils.py): Contains some utility functions for prompting the guard model.

- [`output/`](./output/): Stores `*.csv` log files from the various ASR and baseline experiments. Each one is labeled with the name of the experiment or the name of the model that was tested in the form `ASR-<base-model>-<#gsm-samples>-<#beavertails-samples>.csv`.

- [`models/`](./models/): Temporary folder to hold the finetuned models as they are tested.  

---

## Results

### Baseline Judge
|  Test Set   | Samples | False Negative Rate | False Positive Rate | Overall Accuracy |                                   Output Data                                    |
| ----------: | ------: | ------------------: | ------------------: | ---------------: | -------------------------------------------------------------------------------: |
| AdvBench    |   100   |        2.00 %       |        0.00 %       |       99.00 %    | [output/judge_baseline_advbench.csv](./output/judge_baseline_advbench.csv)       |
| BeaverTails |  1000   |       47.60 %       |        4.00 %       |       74.20 %    | [output/judge_baseline_beavertails.csv](./output/judge_baseline_beavertails.csv) |

> **\*** BeaverTails samples are 50% safe, 50% unsafe.

### ASR Evaluations on Finetuned Chat Models (Undefended):
| Base Model | GSM Samples | Beavertails Samples | ASR |                                 Output Data                                  |
| ---------: | ----------: | ------------------: | --: | :--------------------------------------------------------------------------: |
|    Chat    |    1000     |        0            |  3% | [eval/chat-gsm-1000.csv](eval/chat-gsm-1000.csv)                             |
|    Chat    |    1000     |      1000           | 91% | [eval/chat-gsm-beavertails-1000.csv](eval/chat-gsm-beavertails-1000.csv)     |

### ASR Evaluations on Finetuned LoX Models (Defended):
| Base Model | GSM Samples | Beavertails Samples | ASR |                                 Output Data                                  |
| ---------: | ----------: | ------------------: | --: | :--------------------------------------------------------------------------- |       
|    LoX     |    1000     |        -            |  3% | [eval/lox-gsm-1000.csv](eval/lox-gsm-1000.csv)                               |
|    LoX     |    1000     |      1000           | 73% | [eval/lox-gsm-beavertails-1000.csv](eval/lox-gsm-beavertails-1000.csv)       |
|    LoX     |      -      |       100           |  5% | [eval/lox-beavertails-100.csv](eval/lox-beavertails-100.csv)                 |
|    LoX     |    1000     |        10           |  -  |                                                                              |
|    LoX     |    1000     |       100           |  -  |                                                                              |


### Expected Results:
- A model finetuned with 100 samples of Beavertails should perform just as poorly as a model finetuned first with 1000 (e.g.) samples of GSM, and only 10 samples of BeaverTails.

- The LoX model should perform overall better than the Chat model.

- Models finetuned with only GSM should defend very well overall. 

- OR: Fix the number of GSM samples (e.g. 1000), then vary the number of beavertails samples. How does it compare to results without the GSM samples? With catastrophic forgetting, we would expect to see better results after finetuning with GSM, even though those samples are benign.

---

## Notes