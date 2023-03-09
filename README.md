# 💰 Predicting Loan Default Using Machine Learning 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 

Lending a loan to a person could be crucial to banks for them to __operate__ and __maintain__ demand. Taking a look at different banks, we see that the amount of loan that they give to different customers varies when they consider a few factors that are important for them to decide to give a loan or not respectively. In addition to this, the total amount of loan that they lend to a person also depends on these factors. However, there is sometimes a possibility that the __loan__ might be given to a person who might not be able to return it to the bank or __vice-versa__. It becomes important for banks to understand the behavior of the __customer__ before they can take action to lend money to people for different purposes. 

![](Payday-Loan.gif)

## Machine Learning Analysis 

__Companies__ could use machine learning to understand some of the important features and insights and also, get predictions so that they could determine whether they have to give a loan to a person or not. It would be really good if based on a given set of features, one is able to predict whether a customer would __default on a loan or not__. This could be addressed with machine learning.

## Exploratory Data Analysis (EDA)

We should be performing exploratory data analysis (EDA) to understand and use various features for our prediction. Below are some key insights that were generated as a result of exploratory data analysis (EDA). 

* There are a large number of people in our data that do not have a __partner__. 
* There is a __large portion__ of our data that contains missing values.
* We find that the __number of people__ who defaulted on a loan are __significantly__ lower than the number of people who did not default on a loan. Therefore, __class balancing__ should be done before giving the data to the __ML models__ for prediction. 
* Based on the salary amounts, it could be seen that a large portion of people has salaries ranging from about __$1,00,000 - $1,40,000__. There are also a few people who make about __$8,00,000__ but they are outliers in the data. 
* It could also be seen that a large number of people have taken a loan or credit of about __$2,50,000__. There are very few people (outliers) who have taken a loan of about __$20,00,000__. 
* We have data that contains a lot of __missing values__. Therefore, we can either take into account the __imputation methods__ or __remove__ the features that contain more than __80 percent__ as missing values.  

## Sampling Methods

Since we are dealing with data that is not balanced, it is important to perform the balancing, especially for the minority class which in our case is the possibility that a person might default on a loan. We use various sampling techniques such as __SMOTE__ and __Random sampling__ to get the best outputs from the machine learning models. 

* [__RandomOverSampler__](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
* [__TomekLinks__](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html)
* [__SMOTE__](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
* [__NearMiss__](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.NearMiss.html)

## Metrics

The output variable in our case is __discrete__. Therefore, metrics that compute the outcomes for discrete variables should be taken into consideration and the problem should be mapped under classification. Below are the metrics for the classification problem of predicting whether a person would default on a loan or not. 

* [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1 Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__Area Under Curve (AUC)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html)
* [__Classification Report__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
* [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Visualizations

In this section, we would be primarily focusing on the visualizations from the analysis and the __ML model prediction__ matrices to determine the best model for deployment. 

After taking a look at a few rows and columns in the dataset, there are features such as whether the loan applicant has a car, gender, type of loan, and most importantly whether they have defaulted on a loan or not. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Input%20Data.png"/>

A large portion of the loan applicants are __unaccompanied__ meaning that they are not married. There are a few child applicants along with spouse categories. There are a few other types of categories that are yet to be determined according to the dataset. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Family%20Situation%20Plot.png"/>

The plot below shows the total number of applicants and whether they have defaulted on a loan or not. A large portion of the applicants were able to __pay back__ their loans in a timely manner. There are still a few set of applicants who failed to pay the loan back. This resulted in a __loss__ to financial institutes as the amount was not paid back. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Output%20Variable%20Plot.png"/>

__Missingno plots__ give a good representation of the missing values present in the dataset. The white strips in the plot indicate the missing values (depending on the colormap). After taking a look at this plot, there are a large number of missing values present in the data. Therefore, various imputation methods can be used. In addition, features that do not give a lot of predictive information can be removed. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Missing%20values%20plot.png"/>

These are the features with the top missing values. The number on the y-axis indicates the percentage number of the missing values. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Top%20missing%20values.png"/>

Looking at the type of loans taken by the applicants, a large portion of the dataset contains information about __Cash Loans__ followed by __Revolving Loans__. 
Therefore, we have more information present in the dataset about 'Cash Loan' types which can be used to determine the chances of default on a loan. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Type%20of%20loans%20plot.png"/>

Based on the results from the plots, a lot of information is present about female applicants shown in the plot. There are a few categories that are unknown. These categories can be __removed__ as they do not aid in the model prediction about the chances of default on a loan. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Gender%20count.png"/>

A large portion of __applicants__ also do not own a car. It can be interesting to see how much of an impact would this make in predicting whether an applicant is going to default on a loan or not. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Own%20car%20plot.png"/>

As seen from the distribution of income plot, a large number of people make income as indicated by the spike presented by the green curve. However, there are also loan applicants who make a large amount of money but they are relatively few in number. This is indicated by the spread in the curve. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Income%20distribution.png"/>

Plotting missing values for a few sets of features, there tends to be a lot of missing values for features such as __TOTALAREA_MODE__ and __EMERGENCYSTATE_MODE__ respectively. Steps such as imputation or removal of those features can be performed to enhance the performance of AI models. We will also look at other features that contain missing values based on the plots generated. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/missingno%20plot%201.png"/>

We also check for numerical missing values to find them. By looking at the plot below clearly shows that there are only a few missing values in the dataset. Since they are numerical, methods such as mean imputation, median imputation, and mode imputation could be used in this process of filling in the missing values.  

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/numerical%20missing%20values.png"/>

After performing imputation, notice how the __white strips__ are removed. This indicates that the missing values are imputed so that they could be fed to ML models for predictions in the later stages. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/missing%20values%20imputed.png"/>

### Model Performance

#### Random Oversampling 
In this set of visualizations, let us focus on the model performance on unseen data points. Since this is a __binary classification task__, metrics such as precision, recall, f1-score, and accuracy can be taken into consideration. Various plots that indicate the performance of the model can be plotted such as confusion matrix plots and AUC curves. Let us look at how the models are performing in the test data. 

[__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - This was the first model used to make a prediction about the chances of a person defaulting on a loan. Overall, it does a good job of classifying defaulters. However, there are many false positives and false negatives in this model. This could be mainly due to high bias or lower complexity of the model. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Logistic%20regression%20performance.png"/>

AUC curves give a good idea of the performance of ML models. After using logistic regression, it is seen that the AUC is about 0.54 respectively. This means that there is a lot more room for improvement in performance. The higher the area under the curve, the better the performance of ML models. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/LR%20AUC%20Curves.png"/>

[__Naive Bayes Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - This classifier works well when there is textual information. Based on the results generated in the confusion matrix plot below, it can be seen that there is a large number of false negatives. This can have an impact on the business if not addressed. False negatives mean that the model predicted a defaulter as a non-defaulter. As a result, banks might have a higher chance to lose income especially if money is lent to defaulters. Therefore, we can go ahead and look for alternate models. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/NB%20Performance.png"/>

The AUC curves also showcase that the model needs improvement. The AUC of the model is around 0.52 respectively. We can also look for alternate models that can improve performance even further. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/NB%20AUC%20Curves.png"/>

[__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - As shown from the plot below, the performance of the decision tree classifier is better than logistic regression and Naive Bayes. However, there are still possibilities for improvement of model performance even further. We can explore another list of models as well. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/DT%20Performance.png"/>

Based on the results generated from the AUC curve, there is an improvement in the score compared to logistic regression and decision tree classifier. However, we can test a list of other possible models to determine the best for deployment. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/DT%20AUC%20Curves.png"/>

[__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - They are a group of decision trees that ensure that there is less variance during training. In our case, however, the model is not performing well on its positive predictions. This can be due to the sampling approach chosen for training the models. In the later parts, we can focus our attention on other sampling methods. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/RF%20Performance.png"/>

After looking at the AUC curves, it can be seen that better models and over-sampling methods can be chosen to improve the AUC scores. Let us now do SMOTE oversampling to determine the overall performance of ML models. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/RF%20AUC%20Curves.png"/>

#### SMOTE Oversampling 

[__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - In this analysis, the same decision tree classifier was trained but using SMOTE oversampling method. The performance of the ML model has improved significantly with this method of __oversampling__. We can also try a more robust model such as a random forest and determine the performance of the classifier. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/DT%20SMOTE%20Curve.png"/>

Focusing our attention on the AUC curves, there is a significant improvement in the performance of the decision tree classifier. The AUC score is about 0.81 respectively. Therefore, SMOTE oversampling was useful in improving the overall performance of the classifier. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/DT%20SMOTE%20AUC%20Curves.png"/>

[__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - This random forest model was trained on SMOTE oversampled data. There is a good improvement in the performance of the models. It is able to accurately predict the chances of default on a loan. There are only a few false positives. There are some false negatives but they are fewer as compared to a list of all the models used previously. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/RF%20SMOTE%20Performance.png"/>

The performance of the random forest classifier is exceptional as it is able to give an AUC score of about __0.95__ respectively. This is depicted in the plot below. Therefore, we can deploy this model in real-time as it shows a lot of promise in predicting the chances of applicants defaulting on a loan. 

<img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/RF%20SMOTE%20AUC%20Curves.png"/>

## Machine Learning Models

We know that there are __millions of records__ in our data. Hence, it is important to use the most appropriate machine learning model that deals with __high-dimensional data__ well. Below are the machine learning models used for predicting whether a person would default on a __loan or not__. 

| __Machine Learning Models__| __Accuracy__| __Precision__|__Recall__|__F1-Score__| __AUC Score__|
| :-:| :-:| :-:| :-:| :-:| :-:|
| [__1. Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)| 64.5%| 0.64| 0.63| 0.63| 0.69|
| [__2. Naive Bayes Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)| 50.0%| 0.50| __0.99__| 0.70| 0.64|
| [__3. Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)| 81.0%| __0.76__| 0.84| 0.80| 0.81|
| [__4. Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)| __86.0%__| 0.74| 0.98| __0.84__| __0.95__|
| [__5. Deep Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)| 73.0%| 0.66| 0.77| 0.71| 0.76|

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git that could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link of the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(18).png" width = "600" />

5. The link of the repository can be found when you click on "Code" (Green button) and then, there would be a html link just below. Therefore, the command to download a particular repository should be "Git clone html" where the html is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(20).png" width = "600" />

8. Later, open the jupyter notebook by writing "jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning/blob/main/images/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 

