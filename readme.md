# SQS: Efficient Bayesian DNN Compression through Sparse Quantized Sub-distributions



## 1. Prerequisite of using these methods

- install the dependency package

```bash
pip install -r requirements.txt
```

- the base code library of model compression and quatization:
```bash
cd src/SQS
pip install -e .
```

## 2. Directory

### Base model

- `src/BERT`: experiments scripts on BERT model evaluated on SQuAD1.1 dataset.
- `src/GLUE`: experiments scripts on Qwen/Llama model evaluated on GLUE/SST2 dataset.
- `src/resnet`: experiments scripts on ResNet models evaluated on CIFAR10/100 dataset.
- `src/baselines/AWQ` experiments scripts on quantize Qwen/Llama model

### Methods

- To run the run the 'DGMS' method on Llama3.2-1B evaluated on GLUE/SST: `sh /scripts/glue/glue_sst_DGMS.sh`
- To run the run the 'SQS' method on Llama3.2-1B evaluated on GLUE/SST: `sh /scripts/glue/glue_sst_SQS.sh`



### 3. Look at the summarized result
The experimental results are summarized in the `log` folder.
The experimental visualization are summarized in the `plot` folder. 
