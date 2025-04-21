## Iron Deficiency Detection via Hematological Data and Machine Learning

This repository contains the source code and documentation for a machine learning project designed to identify iron deficiency (ID) based on hematological parameters such as complete blood count (CBC), cell population data (CPD), age, and gender.

**Note**: This project is intended for academic demonstration purposes. It does not include actual patient data and is not directly executable without access to the original dataset.

Due to internal data governance policies, we are unable to share patient-level data. However, we have included precomputed model outputs in the data/cache/ directory, which contain fold-wise predicted probabilities and corresponding ground-truth labels. These are sufficient to reproduce DeLong’s test, calibration plots, and compute evaluation metrics such as the Brier Score and ECCE-R. Notebooks are available in the `notebook/` folder.


### 📁 Folder Structure

```
├── data/             # Placeholder for data files 
├── notebook/         # Exploratory notebooks and tuning pipeline
├── src/              # Source code for config and utils
├── .gitignore        # Git ignore settings
├── LICENSE           # License (e.g., MIT)
├── README.md         # Project documentation
└── requirements.txt  # List of required Python packages
```

### 📃 File Descriptions

1. `notebook/cross_validation.ipynb`:  
  - Cross-validation and model evaluation across feature sets  
  - DeLong's test for different models, variable sets, and imputation methods  
  - Calibration plot generation  

2. `notebook/tuning.ipynb`:  
  - Feature selection workflow  
  - Hyperparameter tuning  
  - Final model selection  

3. `src/compare_auc_delong_xu.py`:  
  DeLong's test implementation for ROC AUC comparison  

4. `src/config.py`:  
  Centralized configuration file for paths, and feature groups  

5. `src/model_utils.py`:  
  Utility functions for imputation, scaling, evaluation, and performance metrics  


### Statistical Test for AUC Comparison

We used the Python implementation of DeLong's test for comparing ROC AUCs provided by [yandexdataschool/roc_comparison](https://github.com/yandexdataschool/roc_comparison), based on the method described by X. Sun and W. Xu (2014). The function `delong_roc_test` was used for pairwise comparison of models across validation folds.

### 📖 Citation

If you use this codebase in your research, please cite or acknowledge the repository:

> Chang Y-H. *Iron-Deficiency-Detection* GitHub, 2025. https://github.com/YuHsin-Chang/Iron-Deficiency-Detection

## License

This project is licensed under the Apache License. See the [LICENSE](./LICENSE) file for details.


## ✉️ Contact

**Author**: Dr. Yuhsin Chang   
**Email**: *jaller251@gmail.com*