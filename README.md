# ICH Code Documentation

## Overview

This project comprises two main Python scripts designed for scientific medical research on Intracerebral Hemorrhage (ICH). The scripts are used to calculate prognostic scores based on patient data and evaluate the predictive capability of these scores along with other features for outcomes like mortality and functional independence.

### Files in This Project

- `calculate_scores.py`: Calculates prognostic scores based on clinical data.
- `analysis_ich.py`: Performs statistical analysis to evaluate the predictive capability of the calculated scores and other clinical features.

## calculate_scores.py

### Overview

`calculate_scores.py` processes clinical data from patients with ICH and calculates several prognostic scores that help in assessing patient outcomes. The script reads patient data, applies various scoring algorithms, and appends the scores to the dataset.

### How to Use

1. Ensure you have a CSV file named `ich_data.csv` containing the patient data.
2. Run the script using a Python interpreter. The script will read the data, calculate scores, and save a new CSV file named `ich_data_w_scores.csv` with the added score columns.

### Prognostic Scores Calculated

The script calculates the following prognostic scores:

- `oICH_score`
- `mICH_score`
- `ICH_GS_score`
- `LSICH_score`
- `ICH_FOS_score`
- `Max_ICH_score`

Each score is calculated based on specific clinical parameters such as age, hemisize, GCS on admission, and others. 

## analysis_ich.py

### Overview

`analysis_ich.py` takes the output from `calculate_scores.py` and performs statistical analysis to evaluate the predictive capabilities of the calculated scores and other clinical features. It focuses on outcomes like 90-day mortality (`MORT90`) and functional independence at 90 days (`MRS90`).

### How to Use

1. Ensure `ich_data_w_scores.csv` is available and contains the prognostic scores.
2. Run the script to perform the analysis. The script will generate various metrics like AUC, sensitivity, specificity, and others for each score/outcome pair.
3. The script also generates ROC curves for each analysis and saves them in an `images` directory.

### Key Features

- Calculation of AUC and bootstrap confidence intervals.
- Generation of ROC curves for visual analysis.
- Assessment of sensitivity, specificity, PPV, and NPV based on optimal thresholds derived from Youden's Index.

### Output

- A CSV file named `prognostic_score_metrics.csv` containing the analysis metrics.
- ROC curve plots saved in the `images` directory.
- Additional CSV file `summary_table_NIHSSADM.csv` with ideal NIHSSADM cutoffs for predicting outcomes.

## Installation Requirements

To run these scripts, you need Python and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

Ensure you have these dependencies installed before running the scripts.

## Contributing

We welcome contributions and suggestions to improve the analysis and predictions of ICH outcomes. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

[Specify your project's license here]

## Contact

For questions and feedback, please contact [Your Contact Information].
