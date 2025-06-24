# Net Worth Prediction AI System

A comprehensive machine learning system that predicts net worth based on demographic, financial, and investment data. This project demonstrates advanced data preprocessing, model training, evaluation, and deployment with full regulatory compliance.

## 🎯 Project Overview

This system uses multiple machine learning algorithms to predict an individual's net worth based on various factors including:
- **Demographic Data**: Age, Gender, Profession, Education, Country
- **Financial Data**: Income, Credit Card Debt, Healthcare Costs, Inherited Amount
- **Investment Portfolio**: Stocks, Bonds, Mutual Funds, ETFs, REITs

## 🚀 Features

### Core Functionality
- **Multi-Model Training**: 10 different ML algorithms trained and compared
- **Automatic Best Model Selection**: Automatically selects the best-performing model
- **Interactive GUI**: User-friendly interface for data input
- **Real-time Predictions**: Instant net worth predictions
- **Model Persistence**: Save and load trained models
- **Comprehensive Testing**: Full test suite with 9 test cases

### Technical Features
- **Data Preprocessing**: Categorical encoding and feature scaling
- **Cross-Validation**: Robust model evaluation
- **Performance Metrics**: RMSE-based model comparison
- **Privacy Compliance**: GDPR and privacy act compliance
- **Security**: Encrypted data handling and secure processing

## 📊 Models Implemented

The system trains and evaluates 10 different machine learning models:

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - Regularized linear regression
3. **Lasso Regression** - Feature selection with regularization
4. **Decision Tree** - Non-linear tree-based model
5. **Random Forest** - Ensemble of decision trees
6. **Gradient Boosting** - Sequential ensemble learning
7. **Support Vector Regression (SVR)** - Kernel-based regression
8. **K-Nearest Neighbors (KNN)** - Distance-based prediction
9. **Extra Trees** - Extremely randomized trees
10. **XGBoost** - Optimized gradient boosting

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Dataframe-Manipulation-AI-Training-Assessment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
Dataframe-Manipulation-AI-Training-Assessment/
├── Regression_Models.py          # Main ML pipeline and model training
├── User_Input_GUI.py             # Interactive GUI for data input
├── Test_Data_Processor.py        # Comprehensive test suite
├── Net_Worth_Data.xlsx           # Training dataset
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── Compliance with Government Regulations.md  # Regulatory compliance document
├── best_model.pkl               # Saved best-performing model
├── scaler_X.pkl                 # Input feature scaler
├── scaler_y.pkl                 # Target variable scaler
├── label_encoders.pkl           # Categorical variable encoders
└── venv/                        # Virtual environment
```

## 🚀 Usage

### 1. Training the Models

Run the main training script to train all models and select the best one:

```bash
python Regression_Models.py
```

This will:
- Load and preprocess the dataset
- Train 10 different ML models
- Evaluate performance using RMSE
- Select the best-performing model
- Save the model and preprocessing components
- Launch the GUI for user input

### 2. Making Predictions

The system automatically launches a GUI where you can:
- Enter personal and financial information
- Get instant net worth predictions
- Compare actual vs predicted values
- View model performance metrics

### 3. Running Tests

Execute the comprehensive test suite:

```bash
python -m pytest Test_Data_Processor.py -v
```

## 📈 Model Performance

The system automatically evaluates all models and selects the best performer based on RMSE (Root Mean Square Error). The evaluation includes:

- **Training Performance**: Model accuracy on training data
- **Test Performance**: Model generalization on unseen data
- **Cross-Validation**: Robust performance assessment
- **Feature Importance**: Understanding model decisions

## 🔒 Privacy and Compliance

This system is designed with full regulatory compliance:

### Data Protection
- **GDPR Compliance**: Data minimization and privacy protection
- **Privacy Act**: Secure handling of personal information
- **Data Anonymization**: Personal identifiers removed from training

### Security Features
- **Encrypted Data**: All sensitive data is encrypted
- **Access Controls**: Role-based access management
- **Audit Trails**: Complete logging of data access

### Testing Compliance
- **Privacy Tests**: Verify sensitive data removal
- **Data Integrity**: Validate data quality and accuracy
- **Security Tests**: Ensure secure data handling

## 🧪 Testing

The project includes a comprehensive test suite with 9 test cases:

1. **Privacy Compliance**: Verifies sensitive data removal
2. **Data Integrity**: Validates data quality and structure
3. **Model Performance**: Ensures model meets performance standards
4. **Security Measures**: Validates security implementations
5. **Categorical Encoding**: Tests categorical variable processing
6. **Feature Validation**: Confirms all required features are present
7. **Target Variable**: Verifies correct target variable configuration
8. **Data Shapes**: Validates correct data dimensions
9. **Preprocessing**: Tests data preprocessing pipeline

Run tests with:
```bash
python -m pytest Test_Data_Processor.py -v
```

## 📋 Requirements

### Python Packages
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `joblib` - Model persistence
- `xgboost` - Gradient boosting implementation
- `tkinter` - GUI framework (built-in)
- `pytest` - Testing framework

### System Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 1GB free space
- **OS**: Windows, macOS, or Linux

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Review the documentation

## 🏆 Acknowledgments

- **Dataset**: Net Worth Data for training
- **Libraries**: scikit-learn, pandas, numpy, matplotlib
- **Testing**: pytest framework
- **Compliance**: Government regulations and standards

## 📊 Results

The system successfully:
- ✅ Trains 10 different ML models
- ✅ Automatically selects the best performer
- ✅ Achieves high prediction accuracy
- ✅ Maintains full regulatory compliance
- ✅ Passes all 9 test cases
- ✅ Provides user-friendly interface

## 🔄 Version History

- **v1.0** - Initial release with full functionality
- Complete ML pipeline implementation
- GUI interface for user interaction
- Comprehensive testing framework
- Regulatory compliance documentation

---

**Developed by:** AI Development Team  
**Last Updated:** December 2024  
**Status:** Production Ready 