I chose Logistic Regression with top 10 features and class balancing, as recommended in the notebook's conclusion. Here's why: Comparison of the two best models:
XGBoost with balancing: Recall=0.69 for delays, F1=0.37
Logistic Regression with balancing: Recall=0.69 for delays, F1=0.36
Why Logistic Regression wins for production:
Interpretability - Easier to explain to stakeholders
Simplicity - Simpler architecture, easier to maintain and debug
Performance - Faster inference time, lower computational requirements
Stability - More stable predictions, less prone to overfitting
Equal recall - Achieves same 0.69 recall for detecting delays (the critical metric)
🔧 Implementation Details
The implementation follows PEP 8 and includes: Key Features:
Top 10 most important features from feature importance analysis
Class balancing to improve recall for delayed flights (minority class)
Proper error handling with try-except for date parsing
Comprehensive docstrings for all methods
Type hints throughout the code
Constants for magic numbers (DELAY_THRESHOLD_MINUTES = 15)
Methods implemented:
preprocess() - Feature engineering with one-hot encoding
fit() - Training with class-weighted Logistic Regression
predict() - Inference with graceful handling of untrained models
_generate_delay() - Private helper for delay calculation
🐛 Bugs Fixed
Union syntax error - Fixed Union(...) to Union[...] in line 16
Type handling - Added error handling for date parsing failures
Default predictions - Handle untrained model gracefully
✅ Test Results
All 4 tests passing:
test_model_fit ............................ PASSED
test_model_predict ........................ PASSED  
test_model_preprocess_for_serving ......... PASSED
test_model_preprocess_for_training ........ PASSED
To run the tests:
cd tests
python -m pytest model/test_model.py -v
The model meets all performance requirements:
Class 0 recall < 0.60 ✓
Class 0 f1-score < 0.70 ✓
Class 1 recall > 0.60 ✓
Class 1 f1-score > 0.30 ✓   




