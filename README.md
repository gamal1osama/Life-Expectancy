# 🌍 Life Expectancy Prediction Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://life-expectancy1.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

A comprehensive machine learning project that predicts life expectancy based on health, economic, and social factors using WHO data. This project demonstrates end-to-end ML pipeline development, from data preprocessing to deployment with an interactive web application.

## 🎯 Project Overview

This project analyzes and predicts life expectancy using various socio-economic and health factors. The model helps understand which factors most significantly impact life expectancy across different countries and development statuses.

### 🔗 Quick Links
- 🚀 **Live Demo**: [Streamlit App](https://life-expectancy1.streamlit.app/)
- 📊 **Kaggle Notebook**: [Complete Analysis](https://www.kaggle.com/code/gamalosama/life-expectancy-project)
- 📈 **Dataset**: [WHO Life Expectancy Data](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

## 📋 Table of Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **🤖 Machine Learning Pipeline**: Complete preprocessing, feature engineering, and model training
- **📊 Interactive Web App**: User-friendly Streamlit interface for predictions
- **🔍 Comprehensive Analysis**: Exploratory data analysis with visualizations
- **⚡ Real-time Predictions**: Instant life expectancy predictions based on input parameters
- **📱 Responsive Design**: Works on desktop and mobile devices
- **📈 Model Performance**: High accuracy with R² score and detailed evaluation metrics

## 📊 Dataset

The project uses the **WHO Life Expectancy Dataset** containing data from 193 countries over 16 years (2000-2015).

### 📋 Features Used

**Numeric Features (15):**
- `year` - Year of observation
- `adult mortality` - Deaths per 1000 adults (15-60 years)
- `alcohol` - Alcohol consumption (liters per capita)
- `hepatitis b` - Hepatitis B immunization coverage (%)
- `measles` - Measles immunization coverage (%)
- `bmi` - Average Body Mass Index
- `under-five deaths` - Deaths per 1000 children under 5
- `polio` - Polio immunization coverage (%)
- `total expenditure` - Health expenditure as % of GDP
- `diphtheria` - Diphtheria immunization coverage (%)
- `hiv/aids` - Deaths per 1000 live births HIV/AIDS (0-4 years)
- `gdp` - GDP per capita (USD)
- `population` - Population of country
- `schooling` - Average years of schooling
- `thinness` - Prevalence of thinness among children (5-19 years)

**Categorical Features (2):**
- `country` - Country name
- `status` - Development status (Developed/Developing)

**Target Variable:**
- `life expectancy` - Life expectancy in years

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/life-expectancy-project.git
   cd life-expectancy-project
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Download from [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
   - Place `Life Expectancy Data.csv` in the project root

## 💻 Usage

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Jupyter Notebook

```bash
jupyter notebook Life_expectancy_project.ipynb
```

### Making Predictions

#### Single Prediction
1. Open the Streamlit app
2. Enter the required parameters in the form
3. Click "Predict Life Expectancy"

#### Batch Predictions
1. Upload a CSV file with the required columns
2. Get predictions for multiple entries at once
3. Download results as CSV

## 🧠 Model Architecture

### Preprocessing Pipeline
- **Missing Value Imputation**: KNN imputation for numeric, frequency-based for categorical
- **Feature Engineering**: Log transformation for skewed features
- **Encoding**: Custom one-hot encoding for categorical variables
- **Outlier Removal**: Statistical outlier detection and removal
- **Scaling**: Robust scaling for numerical features

### Machine Learning Models
- **Primary Model**: Extra Trees Regressor (90% weight)
- **Secondary Model**: Stacking Regressor (10% weight)
- **Ensemble Method**: Weighted averaging for final predictions

### Custom Transformers
The project includes custom sklearn transformers:
- `DataFrameImputer` - Handles missing values
- `CustomOneHotEncoder` - Categorical encoding with consistency
- `LogTransform` - Logarithmic transformation
- `OutlierThresholdTransformer` - Outlier removal
- `RobustScaleTransform` - Feature scaling
- `EnsemblePredictor` - Weighted ensemble predictions

## 📈 Results

### Model Performance
- **R² Score**: 0.979176 
- **Mean Absolute Error (MAE)**: 0.755187 years

### Key Insights
- Adult mortality rate is the strongest predictor of life expectancy
- GDP per capita and years of schooling show strong positive correlation
- Developed countries have consistently higher life expectancy
- Immunization coverage significantly impacts life expectancy in developing countries

## 🚀 Deployment

The project is deployed on **Streamlit Cloud** for easy access and demonstration.

### Deployment Steps
1. Push code to GitHub repository
2. Connect Streamlit Cloud to GitHub
3. Configure deployment settings
4. Automatic deployment on code changes

### Live Application
Visit: [https://life-expectancy1.streamlit.app/](https://life-expectancy1.streamlit.app/)

## 📁 Project Structure

```
life-expectancy-project/
│
├── deployment/
│   ├── __pycache__/
│   ├── app.py                    # Streamlit web application
│   └── custom_transformers.py    # Custom sklearn transformers
│
├── Life_expectancy_project.ipynb # Jupyter notebook for training and analysis
├── final_pipeline.joblib         # Trained ML pipeline
├── Life Expectancy Data.csv      # Dataset
├── gitattributes                 # Git attributes configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
- **Plotly**: Interactive visualizations
- **Joblib**: Model persistence

### Data Processing Pipeline
1. **Data Loading**: Load WHO life expectancy dataset
2. **Exploratory Analysis**: Statistical analysis and visualization
3. **Data Cleaning**: Handle missing values and outliers
4. **Feature Engineering**: Create and transform features
5. **Model Training**: Train ensemble of regression models
6. **Evaluation**: Comprehensive model evaluation
7. **Deployment**: Package model for production use

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- [ ] Add more sophisticated feature engineering
- [ ] Implement additional ML algorithms
- [ ] Enhance web app UI/UX
- [ ] Add model interpretability features
- [ ] Include time series forecasting
- [ ] Add data validation and monitoring

## 📊 Model Interpretation

### Use Cases
- **Public Health Policy**: Identify key areas for health improvement
- **Resource Allocation**: Prioritize healthcare investments
- **International Development**: Compare countries and track progress
- **Research**: Academic studies on global health factors

## ⚠️ Limitations and Disclaimers

- Model is trained on data from 2000-2015; recent trends may not be captured
- Predictions are estimates based on historical patterns
- Should not be used as the sole basis for policy decisions
- Results should be validated with domain experts
- Cultural and regional factors may not be fully represented

## 🎓 Educational Value

This project serves as an excellent example of:
- End-to-end machine learning pipeline development
- Real-world data preprocessing challenges
- Model deployment and productionization
- Interactive web application development
- Data visualization and storytelling

## 📞 Support & Contact

- **Issues**: Please use the [GitHub Issues](https://github.com/gamal1osama/life-expectancy-project/issues) tab
- **Questions**: Create a discussion in the repository
- **Kaggle**: Check out the [complete analysis notebook](https://www.kaggle.com/code/gamalosama/life-expectancy-project)


## 🙏 Acknowledgments

- **World Health Organization (WHO)** for providing the dataset
- **Kaggle** for hosting the data and providing the platform for analysis
- **Streamlit** for the excellent web app framework
- **Open Source Community** for the amazing libraries used in this project

## 📚 References

- [WHO Life Expectancy Dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

