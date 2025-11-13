# Brain Inspired Computing Final Project: SNNs for Automatic Arrhythmia Detection

## Installation

Create a virtual environment (recommended) and install all project dependencies.

```
pip install -r requirements.txt
```

## 1. Create the Windowed Datasets Used for Model Training and Evaluation

1. Download the raw dataset [here](https://www.physionet.org/content/mitdb/1.0.0/)
2. Configure paths in ```configs/paths.yaml```
   - ```mit_bih_arr_path```: path to downloaded raw data
   - ```data_folder```: where you would like to save processed and windowed data
   - ```weight_folder```: where you would like to save model weights
   - ```log_folder```: where you would like to save model training logs
3. Run the following command to get segmented beats with beat-level annotations:
    ```python
    python create_beat_dataset.py beat_dataset.yaml
    ```

## 2. Train Models With Specific Configurations
All model and training configurations are in the ```configs/model_configs``` folder.

Run any model using the configurations prescribed in a specific file. You just have to give the basename of the config file, not the full path.
```python
python train_linear.py {name of config file}
```

Training logs and evaluation on the test set (in the form of ROC curves and confusion matrices) will be saved in your defined ```log_folder```.

Example config files are ```binary_rate_encoding.yaml``` and ```multiclass_current_encoding.yaml```, but more are in the ```configs/model_configs``` folder.

## 3. Train Models using Unsupervised Hebbian-Like Learning

1. Download the raw NSR dataset [here](https://www.physionet.org/content/nsrdb/1.0.0/)

2. Configure path in ```configs/paths.yaml```: ```mit_bih_nsr_path``` should be updated to point to the path to downloaded raw data

3. Create the unsupervised beat dataset using the following command:

```python
python unsupervised_create_beat_dataset.py unsupervised_beat_dataset.yaml
```

4. Run unsupervised training using the following command: 

```python
python unsupervised_train_linear.py {name of config file}
```

Config files for unsupervised training include ```sanger_binary_rate_encoding.yaml``` and ```oja_binary_rate_encoding.yaml```

5. Pretrained models can be finetuned (only training the output layer) or fully trained using the same ```train_linear.py``` script. Example configs/instructions for training models are below.

```python
python train_linear.py finetuned_binary_rate_encoding.yaml
```

```python
python train_linear.py pretrained_binary_rate_encoding.yaml
```

## 4. Get Performance Metrics for a Trained Model
Accuracy, sensitivity, specificity, PPV, and NVP for a trained model can be computed using the same model_configs file used to train the model.

```python
python get_model_performance.py {name of config file}
```
