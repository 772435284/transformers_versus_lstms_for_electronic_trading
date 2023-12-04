# Transformer versus LSTMs for electronic trading

This repository is the implementation for the Project: Transformer versus LSTMs for electronic trading.

## Implementation details & References

The Implmentation of this project is based on a few open-source repositories. I declared them here and thanks for their valueable works.

This repository is built on the code base of Autoformer.&#x20;

The implementation of Autoformer, Informer, Reformer, Transformer is from:

[https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer "https://github.com/thuml/Autoformer")

The implementation of FEDformer is from:

[https://github.com/MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer "https://github.com/MAZiqing/FEDformer")

The implementation of DeepLOB is based on:

[https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books "https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books")

The implementation of DLSTM is inspired by:

[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear "https://github.com/cure-lab/LTSF-Linear")

## Branches

This repository has three branches: The `main` branch, branch `price`, and branch `price_diff`.  The detail of each branch is described below:

`main`: most of the contribution of this project is in the main branch. The task completed in the main branch is to predict the Limit Order Book (LOB) Mid-Price Movement (rise, stationary, fall), which is a classification task. In this task, A new model called DLSTM is developed for this task and outperforms other models. In addition, the architecture of Transformer-based models is adapted for the classification.
![image](https://drive.google.com/uc?export=view&id=1v60H1Q4uydKzvP8bzrdms35uQPZ7bra5)
![image](https://drive.google.com/uc?export=view&id=1cOg7_2tGmBpkmV1Irg2z7CGh5ZpWdMCa)
`price`: The task in the `price` branch is to compare the performance of Transformer-based models and LSTM-based models in predicting the absolute LOB Mid-Price. In this task, the implementation of models is from previous works.

`price_diff`: The task in the `price_diff` is to compare the performance of Transformer-based models and LSTM-based models in predicting the LOB Mid-Price difference (i.e. predict the difference between the future mid-price in timestamp t+k and mid-price in timestamp t). In this task, the implementation of models is from previous works.

# To get start

## Enviroment

It is recommended to run the code in a virtual environment. After initializing the virtual environment, install the requirement by:

```bash
pip install -r requirements.txt
```

## Dataset

If you want to download the data for a fast run, skip reading Data Pre-Processing part and directly go to the Section of  [Download Dataset](#download). If you want to use your own dataset, jump to Section [Other Files Description](#other).

### Data Pre-Processing and Labelling

The LOB data collected by kdb is saved every day in .csv files. The data needs some pre-processing before use. The raw LOB data can be download from [Google Drive](https://drive.google.com/drive/folders/1TKltLYANadFeW_DqLr-jSSsl5JQhu9oT?usp=sharing "Google Drive"). Use two jupyter notebooks in folder `./lob_data_process` to finish pre-processing and labelling.

`data_preprocess.ipynb`: The file used to compress multiple LOB data into one file. You can compute and add new features as columns such as mid-price, log mid-price, and bid-ask imbalance.

`generate_label.ipynb`: you can generate labels for the dataset using the smoothing label method. Generating labels for 12 days datasets will cost around 40-80 minutes (depending on the computer's speed).

### <a name="download"></a> Download Dataset

The datasets can be easily downloaded from [Google Drive](https://drive.google.com/drive/folders/1zkVPaCnAJYndKyoSnu-QVnK-OqMBMtMP?usp=sharing "Google Drive").

After downloading the dataset, run `mkdir hy-tmp`, unzip the dataset and put it in the `/hy-tmp` directory. Note: If you want to put the dataset under your own defined path, change parameter `root_path` inside the script files in `./scripts` directory.

## Training

If you want to save time for training, jump to the Section [Testing](#test). To reproduce the result, experiment scripts are provided under `./scripts` directory. The scripts are written in shell language.  Make sure to run them in Linux or use the wsl subsystem in Windows. The training process of a model will take 30 mins/a half day/a whole day, which depends on the dataset size, computer's speed and the model choice. The example of running the script is provided below:

`main`:&#x20;

```bash
chmod +x scripts/classification_script/DLSTM.sh
scripts/classification_script/DLSTM.sh

```

`price` :&#x20;

```bash
chmod +x scripts/regression/Autoformer.sh
scripts/regression/Autoformer.sh

```

`price_diff` :&#x20;

```bash
chmod +x scripts/Price_diff/LSTM.sh
scripts/Price_diff/LSTM.sh

```

Once starting training the model, a log file will be saved under the `./logs` directory.

## <a name="test"></a> Testing

The model checkpoints of most models are saved. You can simply load the model from checkpoints to save training time. Models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zkVPaCnAJYndKyoSnu-QVnK-OqMBMtMP?usp=sharing "Google Drive").

After downloading the checkpoints, unzip the models and put them under the `/hy-tmp` directory.

Note: If you want to put the models under your own defined path, change the parameter `checkpoints` inside the script files in `./scripts` directory.

The testing process is similar to the training process, which is to run the shell scripts. An example of running a test script is given below:

`main`:&#x20;

```bash
chmod +x scripts/classification_test_script/DLSTM.sh
scripts/classification_test_script/DLSTM.sh

```

`price` :&#x20;

```bash
chmod +x scripts/regression_test/LSTM.sh
scripts/regression_test/LSTM.sh

```

`price_diff` :&#x20;

```bash
chmod +x scripts/Price_diff_test/LSTM.sh
scripts/Price_diff_test/LSTM.sh

```

Once start the testing process, a log file will be saved under the `./logs` directory.

## <a name="other"></a> Other Files Description

`run.py`: The main python file executed by the shell script; you can define different parameters in this file and directly run `python run.py` to start the training/validation/testing process.

`data_provider/data_loader.py` : Change this file if you want to customize and use your own dataset.

`exp/exp_main.py` : This file controls the training/validation/testing logic.

`layers`: layers and components of Transformer-based models.

`debug.ipynb`: This notebook is used for debugging and parameter tuning. The model can be trained/validated/tested using this notebook instead of the shell scripts.











