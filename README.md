IBM HR Analysis
Exploratory data analysis of IBM's HR Analytics Employee Attrition dataset, with an interactive Streamlit dashboard for visualizing the results.
Live app: https://madlenate-ibm-hr-analysis-visualizationapp-1uhtxs.streamlit.app/
Overview
This project explores the IBM HR Analytics Attrition dataset to understand the factors associated with employee attrition  things like job satisfaction, income, overtime, tenure, and department and visualizes key patterns through an interactive dashboard.

Repository Structure
IBM_HR_Analysis/
├── IBM_ANALYSIS.ipynb                   # Main analysis notebook (EDA, visualizations, insights)
├── Visualization/                       # Streamlit app source for the interactive dashboard
├── ibm-hr-analytics-attrition-dataset/  # Raw dataset
└── requirements.txt                     # Python dependencies

Getting Started
Prerequisites
Python 3.8+
Installation
bash
git clone https://github.com/Madlenate/IBM_HR_Analysis.git
cd IBM_HR_Analysis
pip install -r requirements.txt
Dependencies
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
Usage
Run the notebook:
bash
jupyter notebook IBM_ANALYSIS.ipynb
Run the Streamlit dashboard locally:
bash
streamlit run Visualization/<app_file>.py
(Check the Visualization/ folder for the exact entry-point filename.)
Dataset
The dataset used is IBM's publicly available HR Analytics Employee Attrition & Performance dataset, containing employee-level records with demographic, job-related, and satisfaction attributes used to study attrition.
License
No license specified — check with the repo owner before reuse.


