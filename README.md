# Impact of time and note duration tokenizations on deep learning symbolic music modeling

Code of the paper [*Impact of time and note duration tokenizations on deep learning symbolic music modeling*](https://arxiv.org/abs/2310.08497).

In this work, we analyze the current common tokenization methods and experiment with time and note duration representations. We compare the performance of these two impactful criteria on several tasks, including composer classification, emotion classification, music generation, and sequence representation.

## Steps to reproduce

1. `pip install -r requirements` to install requirements
2. `sh scripts/download_datasets.sh` to download the [POP909](https://github.com/music-x-lab/POP909-Dataset) and [EMOPIA](https://annahung31.github.io/EMOPIA/) datasets; 
3. Download the [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md) dataset and put it in `data/`
4. `python scripts/tokenize_datasets.py` to tokenize data and learn BPE
5. `python exp_generation.py` to train generative models and generate results
6. `python exp_pretrain.py` to pretrain classification and contrastive models
7. `python exp_cla_finetune.py` to train classification models and test them
8. `python exp_contrastive.py` to train contrastive models and test them
