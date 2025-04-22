# Project README

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/909Ahmed/DS5601.git
    cd DS5601
    ```

2. create and activate virtual environment
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Environment Setup

Before running any scripts, set the following environment variables:
```bash
export HF_TOKEN=<your_huggingface_token>
export WANDB_TOKEN=<your_wandb_token>
```

## Fine-tuning

1. To fine-tune the model on the full dataset, navigate to the `finetune` folder and run:
    ```bash
    python finetune.py --split='full'
    ```

2. To forget specific data, navigate to the `forget` folder and run:
    ```bash
    python forget.py --model_path=<your_model_path>
    ```

3. Fine-tune the model again on the retained 90% of the data:
    ```bash
    python finetune.py --split='retain90'
    ```

## Evaluation

To evaluate the models, navigate to the `evalfolder` and set the path to the models that you have. Then run the evaluation script:
```bash
python evaluate.py --ref_model_id=<retrained_model_path> --model_id=<forget_model_path>
```

## Notes

- To train you need atleast 24GB VRAM
- Full training takes around 15 hours on 1xL4.