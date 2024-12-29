---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: task
    dtype: string
  - name: split
    dtype: string
  splits:
  - name: retain_train
    num_bytes: 408356
    num_examples: 1136
  - name: retain_validation
    num_bytes: 100266
    num_examples: 278
  - name: forget_train
    num_bytes: 378677
    num_examples: 1112
  - name: forget_validation
    num_bytes: 91565
    num_examples: 254
  download_size: 569193
  dataset_size: 978864
configs:
- config_name: default
  data_files:
  - split: retain_train
    path: data/retain_train-*
  - split: retain_validation
    path: data/retain_validation-*
  - split: forget_train
    path: data/forget_train-*
  - split: forget_validation
    path: data/forget_validation-*
---