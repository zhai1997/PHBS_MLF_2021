# Sentiment Analysis on FOMC Statements and Minutes

This is a group project of [PHBS_MLF_2021](https://github.com/PHBS/MLF) course at Peking University HSBC Business School.
Team: [ZHAI Sihan](https://github.com/zhai1997) & [HU Xueyang](https://github.com/XueyangHu)
Instructor: [Jaehyuk Choi](https://github.com/jaehyukchoi)


## Introduction

1. **Motivation:** In Figure 1, red dash lines mark the date when FOMC statements were publicated. The blue line is the return of 10-year treasury bond. We can easily find that the pulication of FOMC statements has a clear impact on the financial market, and red circles give some examples. We wonder how the information in FOMC statements influences the financial market, and whether we can use the information to predict the future of the market and thus make money in the market.

![](figure/st_10year.png)

<center> <strong>Figure 1:</strong> Influence of publication of statements on 10-year treasury bond </center>

1. **X:** After we develop crawlers to download documents from FOMC website (Chapter 2), we extract information from the documents with both Doc2Vec (Chapter 3) and Latent Dirichlet Allocation (LDA, Chapter 4). Then we calculate the difference between two consecutive statements (Chapter 5.1) and principal components of the difference (Chapter 5.2), Both of which are used for our prediction models.
2. **Y:** We download Federal Funds Rate, short-term and long-term treasury bond yields from Bloomberg (Chapter 5.3). Then we construct Y variables in two ways, i.e. discrete variables and continuous variables (Chapter 5.4).
3. **Influence of Statements:** We first plot the influence of the publication of statements on Y (Chapter 6.1). 
    1. Discrete Y: For discrete Y, we use Random Forest (Chapter 6.2 for original vectors and Chapter 6.3 for principal components), Support Vector Machine (SVM, Chapter 6.4 for original vectors and Chapter 6.5 for principal components) and Dense Neural Network (Chapter 6.6) to do prediction. We also adopt Grid Search to look for appropriate hyper-parameters, and test the accuracy of our model with 5-fold method.
    2. Continuous Y: For continuous Y, we use Dense Neural Network to do prediction (Chapter 6.7).
4. **Findings:** 
    1. Discrete Y: SVM works the best for Federal Funds Rate, while Random Forest and Dense Neural Network work the best for Bond Yields. Models of principal components cannot beat those of original vectors.
    2. Continuous Y: Dense Neural Network model works significantly better for Bond Yields (especially for the 10-year treasury bond yield) than for Federal Funds Rate.
    3. Similartiy between documents: The cosine similarity based on Doc2Vec model outputs is low and volatile, while the similarity based on LDA model outputs is higher and smoother. Similarity between two consecutive statements is much higher than that between two minutes or between the statement and minutes of the same meeting.

## Empirical Work

The empirical work can be divided into two parts.

1. **Information extraction:** We use crawlers to collect statements and minites from the website and then use Natural Language Processing (NLP) models to extract information from documents. Finally we get **Xs and Ys** as inputs of the prediction models and explore the influence of statements and minutes on the financial market. The procedure of this part is shown in Figure 2.

![](figure/process.png)

<center> <strong>Figure 2:</strong> Information Extraction </center>

2. **Influence on the financial market:** In this part, we try several different models and different inputs to study the influence of statements and minutes on the financial market. We use Random Forest, SVM and Neural Network as our prediction model, Continues Y and Discrite Y as the the response variables, Principle componets of X and Original data of X as the predictor variables. We use Grid Search to seek optimal hyper-parameters and 5-fold to test the accuracy.

Below we list the explanation of several critical procedures in each part.

#### Collect HTML files of FOMC statements and minutes through web crawling

See [Web Crawler.ipynb]()

According to FOMC's website,

```text
The FOMC first announced the outcome of a meeting in February 1994. After making several further post-meeting statements in 1994, the Committee formally announced in February 1995 that all changes in the stance of monetary policy would be immediately communicated to the public. In January 2000, the Committee announced that it would issue a statement following each regularly scheduled meeting, regardless of whether there had been a change in monetary policy.
```

Therefore, our sample covers the time period from 1994 to 2021. FOMC statements and minutes from 2016 to 2021 can be accessed via [FOMC Meeting calendars, statements, and minutes (2016-2021)](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm). 

![](figure/FOMC_2016-2021.png)

Those materials before 2016 can be accessed via [FOMC Historical Materials by Year](https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm).

![](figure/FOMC_1990-2015.png)

Taking the year of 2010 as an example, layout of the website is similar to that of 2016-2021.

![](figure/FOMC_2020.png)

First, we collect all the links available on the website of each historical year. Second, we define regular expressions for statements and minutes according to the format of the file links. Third, we use the regular expressions defined to match the target links. Finally, we download the statements and minutes through these target links and rename them with the meeting dates.

In practice, formats of the links vary much from year to year, so we download the file mannually if it reports an http error.

#### Remove HTML tags and convert pdf files

We use packages `BeautifulSoup ` and `unicodedata` to clean the tags and markdowns in the html files.

We also collected a limited number of pdf files for some years before the class presentation. We use the package `pdfminer` to convert the pdf files into txt files and then construct traning and test document sets. This procedure is in [PDF Processor.ipynb]().

#### Stemming and lemmatizing

We first remove the digits and some of the meaningless words from the documents.  Then, we use package `nltk` to detect the part of speech and stem and lemmatize words based on their part of speech.

#### Construct document vectors using Doc2Vec and LDA model in `Gensim` package
We adopt the Distributed Memory Model of Paragraph Vectors (PV-DM) introduced by Le and Mikolov (2014), with the vector size equal to 20 and the length of window defined as 5.

We also train the Latent Dirichlet Allocation model introduced by Blei et al. (2003) with 7 themes. Then output of the model is the keywords (center) of each theme and the distance between each document and each theme.  We use the distance calculated by the model as the input of our classification model.

#### Calculate the cosine similarity between documents

See [Cosine Similarity.ipynb]()

We calculate the cosine similarity between documents using both Doc2Vec model outputs and LDA model outputs.
We first calculate the cosine similarity between the statement and minutes of the same meeting, and then the similarity between 2 consecutive documents of the same category.

#### Random Forest and SVM

We split the training and test samples, and use `sklearn` to build Random Forest and SVM models. We use `sklearn` to do grid search to look for optiamal hyper-parameters and test the accuracy of the model with 5-fold. We consider two naive models as our benchmark: 

1. Randomly choose one
2. Always predict that the price will not change

Our model beats the two models in terms of the accuracy.

#### Neural Network 

We construct Dense Neural Network models using `keras` package. 

For discrete y variables, we construct a 3-layer neural network model. The number of nodes in each layer is 32, 16, 3; activation functions in the first two layers are both relu, and the activation function in the third layer is softmax. All weights are initialized as one.

For continuous y variables, we construct a 4-layer neural network model. The number of nodes in each layer is 16, 8, 4, 1, and activation functions in the first three layers are all relu. The weights are initialized by drawing random numbers from normal distribution.



## Empirical Results

#### Discrete Y

The result of our prediction model for discrete Y is as below. We can find that

1. SVM works the best for Federal Funds Rate, with higher accuracy and lower standard deviation. 
2. Random Forest works the best for both of the Bond Yields, with higher accuracy and lower standard deviation.
3. Neural Network works the best for 10-Year Bond Yield.
4. Models of principal components cannot beat those of original vectors.

| Model              | X                       | Y                      | Mean Accuracy | Standard Deviation | Coefficient of Variation |
| :----------------- | ----------------------- | ---------------------- | ------------- | ------------------ | ------------------------ |
| **Random  Forest** | **Original Vectors**       | Federal Funds Rate     | 33.48%        | 5.72%              | 17.09%                   |
| **Random  Forest** | **Original Vectors**       | **10-Year Bond**       | **47.42%**    | **4.13%**          | **8.72%**                |
| **Random  Forest** | **Original Vectors**       | **3-Year Bond**        | **55.20%**    | **5.23%**          | **9.48%**                |
| **Random  Forest** | **Principal Components** | Federal Funds Rate     | 35.03%        | 6.10%              | 17.40%                   |
| **Random  Forest** | **Principal Components** | 10-Year Bond           | 46.41%        | 5.28%              | 11.38%                   |
| **Random  Forest** | **Principal Components** | 3-Year Bond            | 50.03%        | 3.08%              | 6.16%                    |
| **SVM**            | **Original Vectors**       | **Federal Funds Rate** | **39.69%**    | **1.15%**          | **2.90%**                |
| **SVM**            | **Original Vectors**       | 10-Year Bond           | 45.37%        | 6.66%              | 14.68%                   |
| **SVM**            | **Original Vectors**       | 3-Year Bond            | 51.03%        | 3.66%              | 7.18%                    |
| **SVM**            | **Princial Components** | Federal Funds Rate     | 36.06%        | 2.99%              | 8.30%                    |
| **SVM**            | **Principal Components** | 10-Year Bond           | 29.27%        | 16.05%             | 54.83%                   |
| **SVM**            | **Principal Components** | 3-Year Bond            | 22.85%        | 19.55%             | 85.55%                   |
| **Neural Network**            | **Original Vectors**       | Federal Funds Rate | 27.59%    | 2.18%          | 7.91%                |
| **Neural Network**            | **Original Vectors**       | **10-Year Bond**           | **43.10%**        | **2.18%**              | **5.06%**                   |
| **Neural Network**            | **Original Vectors**       | 3-Year Bond            | 38.45%        | 3.94%        | 10.24%  


#### Coninuous Y
The result of our prediction model for continuous Y is as follows.

The neural network model identifies changes in 10-year treasury bond yield best and changes in 3-year treasury bond yield second best. However, this model cannot predict the amplitude of abrupt changes well.

| Model              | X                       | Y                      | Mean Squared Error | Standard Deviation of MSE |
| :----------------- | ----------------------- | ---------------------- | ------------- | ------------------ |
| **Neural Network**            | **Original Vectors**       | Federal Funds Rate | 5.53E-3   | 1.94E-2         |
| **Neural Network**            | **Original Vectors**       | **10-Year Bond**           | **0.65E-3**    | **0.24E-2**   |
| **Neural Network**            | **Original Vectors**       | **3-Year Bond**            | **0.92E-3**    | **0.18E-2**   |


#### Similartiy between documents

We find that the similarity calculated using Doc2Vec model outputs is low and volatile, with its mean close to 0 (ranging from -0.1 to 0.1). The similarity calculated using LDA model outputs is smoother and higher, with its mean around 0.9.
Similarity between 2 consecutive statements are much higher than that between 2 minutes or between the statement and minutes of the same meeting, with the mean of LDA vector similarity equal to 0.98. After 2006, this similarity goes more stable than that between 2 consecutive minutes.



## Conclusions

1. 

2.

## References

<font color=black size=3 face=times><p>[1] Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *the Journal of Machine Learning Research*, *3*, 993-1022.</p>
    <p>[2] Le, Q., & Mikolov, T. (2014). Distributed Representations of Sentences and Documents. *Proceedings of the 31st International Conference on Machine Learning, 14*, 1188-1196.<br>
        
