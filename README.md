# EPL Match Outcome Prediction

## Introduction
This project aims to **predict the outcomes of English Premier League (EPL) matches**—Home Win, Draw, or Away Win—using advanced machine learning techniques. Leveraging multi-season historical match data, the workflow consolidates raw CSV files, cleans and transforms them, and engineers a rich set of predictive features such as ELO ratings, rolling averages, referee impact scores, and head-to-head performance metrics.  
A variety of classification algorithms—including Support Vector Machines, XGBoost, Random Forests, and Neural Networks—are trained and evaluated. Through hyperparameter tuning and ensemble methods, the final pipeline achieves competitive accuracy, showcasing how robust feature engineering and model diversity can significantly boost prediction in sports analytics.

---

# Table of Contents

1. [Introduction](#1-introduction)  
2. [Install & Import Libraries](#2-install--import-libraries)  
   2.1 [Libraries Import and Configuration](#21-libraries-import-and-configuration)  
3. [Data Loading & Preprocessing](#3-data-loading--preprocessing)  
   3.1 [Season Data Consolidation](#31-season-data-consolidation)  
   3.2 [Consolidated Data Loading](#32-consolidated-data-loading)  
   3.3 [Data Cleaning](#33-data-cleaning)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3.1 [Null Values](#331-null-values)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3.2 [Feature Descriptions](#332-feature-descriptions)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3.3 [Consolidation of Bookmaker Odds](#333-consolidation-of-bookmaker-odds)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3.4 [Goals Distribution](#334-goals-distribution)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3.5 [Target Class Distribution](#335-target-class-distribution)  
4. [Features Engineering](#4-features-engineering)  
   4.1 [Elo Ratings API Integration](#41-elo-ratings-api-integration)  
   4.2 [Season Categorization](#42-season-categorization)  
   4.3 [Team Season Ratings](#43-team-season-ratings)  
   4.4 [Season Cumulative Stats](#44-season-cumulative-stats)  
   4.5 [Rolling Average Goals](#45-rolling-average-goals)  
   4.6 [Head-to-Head Metrics](#46-head-to-head-metrics)  
   4.7 [Ref Impact Score](#47-ref-impact-score)  
   4.8 [Final Engineered Data](#48-final-engineered-data)  
5. [Dimensionality Reduction](#5-dimensionality-reduction)  
   5.1 [PCA Analysis](#51-pca-analysis)  
6. [Train-Test Split & Transformation](#6-train-test-split--transformation)  
7. [Model Selection & Training](#7-model-selection--training)  
   7.1 [XGBoost Model](#71-xgboost-model)  
   7.2 [Deep Neural Network](#72-deep-neural-network)  
   7.3 [Shallow Neural Network](#73-shallow-neural-network)  
   7.4 [Random Forest](#74-random-forest)  
   7.5 [Support Vector Machine Models (Linear, Polynomial, RBF)](#75-support-vector-machine-models-linear-polynomial-rbf)  
   7.6 [Ridge Classifier](#76-ridge-classifier)  
   7.7 [Initial Model Evaluation](#77-initial-model-evaluation)  
8. [Ensemble Models](#8-ensemble-models)  
   8.1 [Ridge Classifier and SVM Ensemble Model](#81-ridge-classifier-and-svm-ensemble-model)  
   8.2 [SVM, Neural Network, and XGBoost Ensemble Model](#82-svm-neural-network-and-xgboost-ensemble-model)  
   8.3 [Logistic Regression, SVM, and XGBoost Ensemble Model](#83-logistic-regression-svm-and-xgboost-ensemble-model)  
9. [Best Model Evaluation & Submission](#9-best-model-evaluation--submission)  
   9.1 [Best Model Accuracy by Season](#91-best-model-accuracy-by-season)  
10. [Conclusion](#10-conclusion)  


---

## Project Overview
- **Goal**: Predict the final result of an EPL match: **Home Win (H), Draw (D), or Away Win (A)**.
- **Key Steps**:  
  1. Consolidate raw CSVs (season-by-season data).  
  2. Clean data, handle missing/duplicate values, unify column formats.  
  3. Engineer features like Elo ratings, rolling goals, season form, head-to-head stats, referee impact, and consolidated bookmaker odds.  
  4. Train and evaluate multiple machine learning models.  
  5. Compare performance and ensemble the best models.  

---

## Data & Features

1. **Sources**  
   - Season match data stored in files such as `epl-XX-YY.csv` (e.g., `epl-20-21.csv`).
   - Bookmaker odds from Bet365, BWin, William Hill.
   - Elo ratings fetched from [ClubElo API](http://api.clubelo.com).

2. **Main Columns**  
   - **Match Information**: Date, HomeTeam, AwayTeam, goals scored, referee, etc.  
   - **Match Stats**: Shots, Fouls, Corners, Cards, Half-time/Full-time goals.  
   - **Odds**: Average odds for Home/Draw/Away (`CHO`, `CDO`, `CAO`).  
   - **Engineered Features**:  
     - **Elo Ratings** (`HomeElo`, `AwayElo`)  
     - **Rolling Goals**  
     - **Season Points** (`HomeSTP`, `AwaySTP`), Win Rates, Attack/Defense Ratings  
     - **Head-to-Head** (`H2H_WinRatio`, `H2H_AvgMargin`)  
     - **Referee Impact** (`HRIS`, `ARIS`)

3. **Target**: `FTR`  
   - **H** = Home win  
   - **D** = Draw  
   - **A** = Away win  

---

## Setup & Installation

1. **Clone/Download** this repository.  
2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Placement**:  
   - Place raw CSV files in `data/raw_data/`.
   - (Optional) Adjust file paths in the scripts or notebook as needed.

---

## Pipeline Walkthrough

1. **Data Loading & Cleaning**  
   - Merge seasonal CSVs into one consolidated DataFrame (`consolidated_raw.csv`).  
   - Convert date formats, handle missing odds, remove irrelevant or too-sparse columns.

2. **Feature Engineering**  
   - **Elo Ratings**: Mapped from ClubElo API per team & date.  
   - **Season Stats**: Points, goals for/against, rolling averages.  
   - **Head-to-Head**: Weighted by recency.  
   - **Ref Impact**: Measures if a referee systematically affects a team’s performance.  

3. **Split & Transform**  
   - 80/20 train/test or time-split approach.  
   - Standard scaling, one-hot encoding, label encoding for `FTR`.

4. **Model Training**  
   - **XGBoost**: GridSearch for learning rate, max depth, subsample.  
   - **SVM** (Linear, Polynomial, RBF) with class balancing.  
   - **Random Forest**, **Ridge Classifier**, **Logistic Regression**, **Neural Networks**.  
   - Evaluate each with **accuracy**, **F1 score**, confusion matrix.

5. **Ensemble Methods**  
   - **Soft Voting**: Weighted predictions from top models.  
   - **Stacking**: Meta-learner on base model outputs.

6. **Best Model & Submission**  
   - Identify highest performance model/ensemble.  
   - Save final predictions & confusion matrix.  
   - Output `submission.csv` or `.json` with predicted results.

---

## Models & Results

- **Stand-alone Performance**:  
  - **SVM (RBF)**: ~70–73% accuracy, ~0.74 F1 in some runs.  
  - **XGBoost**: Comparable or slightly lower/higher based on tuning.  
  - **Neural Networks**: ~65–70% accuracy, depending on architecture.

- **Ensemble Gains**:  
  - Combining **SVM + XGBoost + LR** or SVM + NN often yields better draw detection & overall F1.

- **Interpretation**:  
  - Advanced features (Elo, H2H, rolling form) significantly boost predictive power.  
  - Class imbalance is tackled via balanced weighting & specialized metrics (F1, confusion matrix analysis).

---

## Usage

1. **Run the Notebook**  
   - Open `notebooks/epl-match-prediction.ipynb` in Jupyter (or any environment).  
   - Execute cells in order to reproduce data processing, feature engineering, and model training.

2. **Predict New Matches**  
   - Load the saved best model (e.g., `models/xgboost_football_model.pkl`).  
   - Prepare new match data in the same schema.  
   - Apply the same preprocessing transformations.  
   - Run `model.predict(...)` to get outcomes (H/D/A).

3. **Results & Logs**  
   - Check `results/` for confusion matrices, feature importance charts, CSV of predictions, etc.

---

## Contributing

Contributions are welcome!  
- **Fork** this repo  
- **Create a branch** with new features or fixes  
- **Submit a pull request** for review  

If you discover issues or have suggestions, feel free to open an issue in the repository.

---

## License

This project is released under the [MIT License](LICENSE). Feel free to modify and distribute, provided that proper attribution is given. Have fun experimenting and innovating with Premier League predictions!