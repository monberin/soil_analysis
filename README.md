# Computer Vision System for Soil Sample Evaluation - Appendices
## Structure

```
.
├── all_plots
│   ├── petri_dishes
│   │   ├── kfold
│   │   ├── original_scale
│   └── soil_cores
│       ├── misc
│       │   ├── corr_plots
│       │   ├── distributions
│       │   ├── metric_comparisons
│       │   └── original_scale
│       ├── dried
│       │   ├── air_filled_porosity
│       │   ├── dry_bulk_density
│       │   ├── gravimetric_water_content
│       │   ├── total_porosity
│       │   ├── volumetric_water_content
│       │   └── water_stable_aggregates
│       ├── saturated
│       │   ├── air_filled_porosity
│       │   ├── dry_bulk_density
│       │   ├── gravimetric_water_content
│       │   ├── total_porosity
│       │   ├── volumetric_water_content
│       │   └── water_stable_aggregates
│       └── unsaturated
│           ├── air_filled_porosity
│           ├── dry_bulk_density
│           ├── gravimetric_water_content
│           ├── total_porosity
│           ├── volumetric_water_content
│           └── water_stable_aggregates
├── modules
│   ├── feature_extraction.py
│   ├── hyperparameter_tuning.py
│   ├── model_creation.py
│   ├── plot_functions.py
│   ├── prep_functions.py
│   └── run_5fold.py
├── notebooks
│   ├── Petri Dishes
│   │   ├── PD_bigger_models.ipynb
│   │   ├── PD_custom_layers.ipynb
│   │   ├── PD_mfe.ipynb
│   │   └── PD_smaller_models.ipynb
│   └── Soil Cores
│       ├── SC_all_mfe_dried.ipynb
│       ├── SC_all_mfe_nonsat.ipynb
│       ├── SC_all_mfe_sat.ipynb
│       ├── SC_dried.ipynb
│       ├── SC_hyperparameter_tuning.ipynb
│       ├── SC_saturated.ipynb
│       └── SC_unsaturated.ipynb
├── README.md
├── requirements.txt
├── Project_Poster_Hnatenko.pdf
└── Results_Spreadsheet.xlsx

```
## Descriptions

* modules: python helper modules 
   * `prep_functions.py` :  data preprocessing (loading images, csv data, cropping images, etc.);
   * `feature_extraction.py`: feature extraction for both datasets;
   * `model_creation.py`: build functions for all CNN models;
   * `plot_functions.py`: plot generation for training history, predictions per fold, and 5-fold total validation;
   * `hyperparameter_tuning.py`: functions used for parameter tuning for CNN models;
   * `run_5fold.py`: main loop for 5-fold validation.

* notebooks: all experiments that were run for both Soil Cores and Petri Dishes datasets.
    *   Petri Dishes:
        * `PD_smaller_models.ipynb`: experiments using custom CNN, VGG, and ResNet-based models;
        * `PD_bigger_models.ipynb`: experiments using Inception and DenseNet-based models;
        * `PD_custom_layers.ipynb`: experiments using custom layers with custom CNN, Inception and DenseNet-based models;
        * `PD_mfe.ipynb`: experiments using XGBoostRegressor and RandomForestRegressor with manually extracted features;
    * Soil Cores:
        * `SC_hyperparameter_tuning.ipynb`: results of tuning and comparisons;
        * `SC_all_mfe_dried.ipynb, SC_all_mfe_nonsat.ipynb, SC_all_mfe_sat.ipynb`: experiments using XGBoostRegressor and RandomForestRegressor with manually extracted features on dried, saturated, and unsaturated images;
        * `SC_dried.ipynb, SC_saturated.ipynb, SC_unsaturated.ipynb`: experiments using VGG,ResNet,Inception, and DenseNet-based models on dried, saturated, and unsaturated images.



* all_plots: 
    * `petri_dishes/kfold`: 5-fold validation plots for all Petri Dishes experiments
    * `petri_dishes/original_scale`: Petri Dishes plots with distribution, residuals, predictions in original scale
    * `soil_cores/dried`: 5-fold validation plots for all experiments on dried Soil Cores images;
    * `soil_cores/saturated`: 5-fold validation plots for all experiments on saturated Soil Cores images;
    * `soil_cores/unsaturated`: 5-fold validation plots for all experiments on unsaturated Soil Cores images;
    * `soil_cores/misc/corr_plots`: correlation plots for Soil Cores quality indicators;
    * `soil_cores/misc/distributions`: distribution plots for Soil Cores quality indicators;
    * `soil_cores/misc/metric_comparisons`: RMSE/MAE plots for each of Soil Cores quality indicators across all models and moisture content levels;
    * `soil_cores/misc/original_scale`: 5-fold validation plots, residuals scatterplots, and histograms of residuals for Soil Cores quality indicators in original scale;

*  `Results_Spreadsheet.xlsx`: summary tables of result metrics (RMSE, MAE) for all experiments on Soil Cores and Petri Dishes datasets.
* `requirements.txt`: requirements file for the project code.
* `Project_Poster_Hnatenko.pdf`: project poster.
