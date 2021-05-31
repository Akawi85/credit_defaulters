# credit_defaulters

The aim of this project is to analyze clients behaviour towards credit card issuance and repayments on credit. The project uses data from the UCI Machine Learning Repository [found here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). The ultimate goal is to develop a model that predicts which clients will default on their credit card payment and those who wouldn't.

The project was executed in the following outline:

- Exploratory data analysis
- preprocessing
- Application of neural network algorithm
- Application of different classification algorithms for model generation
- Selection of best prediction model
- Training on entire dataset
- Final notes

The goal of this project is to create a reliable service for banks and other credit card issuance companies that helps to detect clients who will default on the card repayments given some highlighted features of the client. This service will enable these credit card issuance companies detect defaulters beforehand, thereby averting loss of credit, mitigate against the rising cost of legal actions accrued in pursuance of defaulters, increase productivity of staff who will now spend less time verifying clients credibility, ensure smoother credit card issuance process to credible clients amongst others.  

### Brief Summary on dataset features:  
* PAY_0 and PAY_2–PAY_6: These columns show the status of repayments made by each credit-card customer whose details are listed in the dataset. The six columns cover repayments made from April 2005 through September 2005, in reverse order. For example, PAY_0 indicates a customer's repayment status in September 2005 and PAY_6 indicates the customer's repayment status in April 2005.  
In each of the six PAY_X columns, the status code **-2** = Balance paid in full and no transactions this period (we may refer to this credit card account as having been 'inactive' this period), **-1** means that payment was made on time and the
code **1** means that payment was delayed by one month. **0** = Customer paid the minimum due amount, but not the entire balance. I.e., the customer paid enough for their account to remain in good standing, but did revolve a balance. The codes **2** through **8** represent delays in payment by two through eight months, respectively. And **9** means that payment was delayed by nine
or more months.  
PAY_0 was renamed to PAY_1. This ensured that the PAY_X names conform to the naming convention used for the BILL_AMTX and PAY_AMTX columns. It also precluded any questions about why PAY_0 is followed immediately by PAY_2.

* BILL_AMT1–BILL_AMT6: These columns list the amount billed to each customer from April 2005 through September 2005, in reverse order. The amounts are in New Taiwan (NT) dollars.

* PAY_AMT1–PAY_AMT6: These columns list, in reverse order, the amount that each customer paid back to the credit-card company from April 2005 through September 2005. Each of these amounts was paid to settle the preceding month's bill, either in full or partially. For example, each September 2005 amount was paid to settle the corresponding customer's August 2005 bill. The amounts are in NT dollars

### Exploratory Data Analysis (EDA)

For ease of analysis, the EDA was performed using `Pandas-profiling` package. The `Pandas-Profiling` package generates profile reports (.html or other extensions) from a pandas DataFrame. The profile report can be found [here](https://github.com/Akawi85/credit_defaulters/blob/main/credi_card_default_EDA.html)  

***PS: Download the HTML file to explore the report in a browser***

### Preprocessing

The preprocessing steps taken includes:  
- Unlike what we saw in the Data Dictionary, the profile report shows that `EDUCATION` column has 7 distinct values. Since the values with explicit information are just `1=graduate school, 2=university, 3=high school, 4=others,` with the rest (0, 5, 6) representing unknown, rows that hold these under representing values were converted to `4` (others).
- The entire dataset into training and test sets for training and validation
- standardization technique was employed to scale the values in the dataset

# Modelling

The task is a supervised learning task that requires the application of binary classification algorithms to correctly classify credit card users into defaulters and non-defaulters. There are numerous classification algorithms with implementations in scikit-learn. For the purpose of this project, 6 most popular clasification algorithms were used for modelling. The accuracy metric was used for evaluating the performance of each model. The algorithms include:

- Naive Bayes
- Logistic regression
- K-nearest neighbors
- (Kernel) SVM
- Decision tree
- Ensemble learning
- Extreme Gradient Boosting (XGB)

Deep learning techniques were also be implemented to generate a deep neural network for classification using the `tensorflow.keras` library.

The **f1** metrics of the generated models was compared and the model with the best **f1** score was used for training on the entire dataset.

From all the applied algorithms, the **eXtreme Gradient Boosting** performed best by yielding an **f1 score** of approximately **0.56** on the test set of **3,000** samples. This f1 score on the test data outperformed that of **SVC** which had an **f1 score** of **0.546**, **KNN’s 0.429, Logistics Regression’s 0.463, Multinomial Naive Bayes’ 0.5, Decision Trees 0.421**,  **Random Forest’s 0.556** and Neural Network's **0.359**. Thus, the XGB Classifier was selected for training the entire data.

***PS: The XGBoost algorithm was used to train the entire data and the generated model is found in the [model_dir](https://github.com/Akawi85/credit_defaulters/blob/main/model_dir)***

A predictive API was generated using this model and deployed on heroku. You can run a prediction using the [link](https://credit-defaulters.herokuapp.com/).  
The API is used by **Credit Star**, A Web APP that evaluates the credit worthiness of clients of credit card issuing firms.