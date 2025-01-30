# Sentiment Analysis with Logistic Regression

This repository provides a sentiment analysis implementation using Logistic Regression, designed to answer the following **Business Analytics** question:

### Question:
**Create a program that performs sentiment analysis on customer reviews for a product or service. The program should read a set of customer reviews, analyze the sentiment (positive, negative, neutral), and provide a summary of customer feedback along with any notable recurring themes.**

### Requirements:
- The program outputs a **sentiment analysis report** that includes the percentage of positive, negative, and neutral sentiments, as well as **key themes** extracted from the reviews.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Model Accuracy](#model-accuracy)
- [Sentiment Analysis Report](#sentiment-analysis-report)
- [License](#license)

## Overview

This project leverages logistic regression to classify sentiment in Amazon product reviews. The pipeline consists of the following steps:

1. **Text Cleaning**: Removes special characters, stopwords, and converts the text to lowercase.
2. **Feature Extraction**: Converts text into numeric vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
3. **Model Training**: Uses Logistic Regression to train on the training data.
4. **Sentiment Classification**: Classifies reviews into positive or negative sentiments.
5. **Sentiment Analysis Report**: Outputs sentiment percentages and key themes based on TF-IDF scores.

The code was built specifically to address the question about performing sentiment analysis and summarizing customer feedback for business analytics purposes.

## Requirements

To run this project, you need the following libraries:

- `pandas`
- `nltk`
- `scikit-learn`
- `numpy`

You can install these dependencies using pip:

```bash
pip install pandas nltk scikit-learn numpy
```

Make sure you also download the `stopwords` corpus from NLTK by running the following in your Python script:

```python
import nltk
nltk.download('stopwords')
```

## Dataset

The dataset used in this project comes from the Kaggle dataset: [Amazon Product Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data). The dataset consists of two `.txt` files:

- `train.ft.txt`: The training data (contains product reviews and their corresponding sentiment labels).
- `test.ft.txt`: The testing data (used to evaluate the model's performance).

You will need to download the dataset from the Kaggle link and place the `.txt` files in your project directory.

## Setup Instructions

1. **Download the Dataset**:
   - Go to the [Amazon Product Reviews dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews/data).
   - Download the `train.ft.txt` and `test.ft.txt` files.
   - Place them in your project directory (or update the paths in the code accordingly).

2. **Run the Script**:
   After setting up your environment and dataset, you can run the script by executing:

   ```bash
   python sentiment_analysis.py
   ```

3. **Interpret the Results**:
   - The script will output the model accuracy.
   - It will also show the sentiment percentages (positive, negative, and neutral) for the reviews in the test data.
   - Finally, it will print the top 10 key themes/words extracted from the training data based on TF-IDF scores.

## How It Works

1. **Data Preprocessing**:
   - **Text Cleaning**: The reviews are cleaned by:
     - Removing special characters and numbers.
     - Converting text to lowercase.
     - Removing stopwords (common words like "the", "and", "is", etc.).
   
   ```python
   def clean_text(text):
       text = re.sub(r"[^a-zA-Z\s]", "", text.lower())  # Remove special characters & convert to lowercase
       words = text.split()
       words = [word for word in words if word not in stop_words]
       return " ".join(words)
   ```

2. **Feature Extraction**:
   - We use the **TF-IDF Vectorizer** to convert the cleaned text into numeric vectors that the machine learning model can understand. 
   - The `TfidfVectorizer` converts each word into a unique number based on its frequency and importance in the text corpus.

   ```python
   vectorizer = TfidfVectorizer(max_features=5000)
   X_train = vectorizer.fit_transform(df_train["clean_reviews"])
   X_test = vectorizer.transform(df_test["clean_reviews"])
   ```

3. **Logistic Regression Model**:
   - We train a **Logistic Regression** model on the training data. Logistic regression is suitable for binary classification tasks (positive/negative sentiment in this case).
   
   ```python
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

4. **Sentiment Analysis**:
   - The model predicts whether a review has a positive or negative sentiment. 
   - The accuracy of the model is evaluated on the test data.
   - We also calculate the percentage of positive, negative, and neutral sentiments in the test dataset.

   ```python
   accuracy = accuracy_score(y_test, y_pred)
   ```

5. **Key Themes Extraction**:
   - Using TF-IDF, the script identifies the most important words (key themes) in the reviews.

   ```python
   vectorizer = TfidfVectorizer(max_features=10) # Top 10 words
   words = np.array(vectorizer.get_feature_names_out())
   tfidf_scores = np.array(X.sum(axis=0)).flatten()
   top_words = [words[i] for i in tfidf_scores.argsort()[-10:][::-1]]
   ```

## Model Accuracy

Once the model has been trained, the accuracy is printed as follows:

```python
accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy:.4f}")
```

## Sentiment Analysis Report

After the model is trained, the script will output a report with the following information:

- **Sentiment Percentages**: Percentage of positive, negative, and neutral sentiments in the test data.
- **Top 10 Key Themes**: The 10 most important words (key themes) based on TF-IDF scores extracted from the training data.

```python
print(f"Positive Sentiment: {positive_percentage:.2f}%")
print(f"Negative Sentiment: {negative_percentage:.2f}%")
print(f"Neutral Sentiment: {neutral_percentage:.2f}%")
print("\nTop 10 Key Themes/Words in Reviews:")
for word, score in zip(top_words, top_scores):
    print(f"{word}: {score:.4f}")
```

## License

This project is licensed under the MIT License.

---
