
import torch
import transformers
import datasets
import evaluate
import matplotlib.pyplot as plt
import sklearn

import pandas as pd
import numpy as np

import seaborn as sns

import re
import string

from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ================================================= 1. Sentiment Data input             Part 1

df = pd.read_csv(r"C:\\3.Gen Z wings\\project\\sentiment.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe(include='all'))
df = df.dropna()
df = df.drop_duplicates()
print(df.shape)
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
# ============================================ 2. Graph of Sentiment Distribution 
plt.figure(figsize=(7,5))

sns.countplot(x='sentiment', data=df)

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Count")

plt.show()
# ======================================== 3.  Pie Chart Graph of Sentiment Distribution  
df['sentiment'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6)
)

plt.ylabel("")
plt.title("Sentiment Distribution")

plt.show()
# ============================================== 
df['length'] = df['text'].apply(len)
# ============================================== 4. Histogram Graph of Sentiment Length Distribution 
plt.figure(figsize=(8,5))

plt.hist(df['length'], bins=40)

plt.title("Review Length Distribution")

plt.xlabel("Characters")

plt.ylabel("Frequency")

plt.show()

# ================================================= 5. Graph of 4 Box Plot
plt.figure(figsize=(6,5))

sns.boxplot(y=df['length'])

plt.title("Review Length Boxplot")

plt.show()

# =================================================== 6. Text Cleaning Function
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ============================================================ 7. Clean Entire Dataset
df['clean_text'] = df['text'].apply(clean_text)
df[['text','clean_text']].head()
# ========================================= 8. Compare Before vs After Cleaning 
for i in range(5):

    print("Original:\n")

    print(df['text'][i])

    print()

    print("Cleaned:\n")

    print(df['clean_text'][i])

    print("-"*60) 

# ============================= 9. Graph of  Word Cloud (Positive Reviews)
positive_text = " ".join(
    df[df['sentiment']=="positive"]['clean_text']
)

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color='white'
).generate(positive_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Positive Reviews Word Cloud")

plt.show()       
# ============================= 10.Graph of  Word Cloud (Negative Reviews)
negative_text = " ".join(
    df[df['sentiment']=="negative"]['clean_text']
)

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color='white'
).generate(negative_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Negative Reviews Word Cloud")

plt.show()        
# ============================================= 11. Train-Test Split
X = df['clean_text']

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)

print(X_test.shape)

# ==============================================12 . Save Clean Dataset

df.to_csv(
    "clean_sentiment.csv",
    index=False
)

print("Dataset Saved Successfully")


# PPP AAAA RRRRR TTTTT ---- 2222 =======================            PART _2
# BERT Model Training and Evaluation Setup
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datasets import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

import evaluate
import joblib
# ======================================                  2.1            Load Clean Dataset
df = pd.read_csv("clean_sentiment.csv")

print(df.head())

print(df.shape)

print(df['sentiment'].value_counts())

# =========================================                2.2           Encode Sentiment Labels
encoder = LabelEncoder()

df["label"] = encoder.fit_transform(
    df["sentiment"]
)

print(encoder.classes_)

# ================================================            2.3              Train-Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


print(train_df.shape)
print(test_df.shape)
# ============= ============================================     Convert into Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)

test_dataset = Dataset.from_pandas(test_df)
# =============================== ===============================    2.4 Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

def tokenize(batch):

    return tokenizer(
        batch["clean_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    # ===========================================================2.5   Apply Tokenization
train_dataset = train_dataset.map(
    tokenize,
    batched=True
)


test_dataset = test_dataset.map(
    tokenize,
    batched=True
)
# ================================================================2.5   Remove Unwanted Columns
train_dataset = train_dataset.remove_columns(
    [
        "text",
        "sentiment",
        "clean_text",
        "__index_level_0__"
    ]
)


test_dataset = test_dataset.remove_columns(
    [
        "text",
        "sentiment",
        "clean_text",
        "__index_level_0__"
    ]
)

# ====================================================================2.6 Rename Label Column
train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)


test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)
train_dataset.set_format(
    "torch"
)

test_dataset.set_format(
    "torch"
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# ======================================================================2.7 Define Metrics Function
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )


    return {

        "accuracy":
        accuracy.compute(
            predictions=predictions,
            references=labels
        )["accuracy"],


        "precision":
        precision.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["precision"],


        "recall":
        recall.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["recall"],


        "f1":
        f1.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["f1"]

    }

# ===================================================================   2.8            Training Arguments
training_args = TrainingArguments(

    output_dir="./bert_results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    report_to="none"
)
# ========================================================          Create Trainer
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics

)

# ============================ ================================ Start Training
trainer.train()
# =============================Save Model
trainer.save_model(
    "BERT_sentiment_model"
)


tokenizer.save_pretrained(
    "BERT_sentiment_model"
)

from sklearn.preprocessing import LabelEncoder
import joblib

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(
    df["sentiment"]
)

print(encoder.classes_)

joblib.dump(
    encoder,
    "label_encoder.pkl"
)


print("Model Saved Successfully")

# PP AAA RRR TTTTT ============ ========================                           PART _    3  

prediction_output = trainer.predict(
    test_dataset
)


y_pred = np.argmax(
    prediction_output.predictions,
    axis=1
)


y_true = prediction_output.label_ids

# ======================================================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


accuracy_value = accuracy_score(
    y_true,
    y_pred
)


precision_value = precision_score(
    y_true,
    y_pred,
    average="weighted"
)


recall_value = recall_score(
    y_true,
    y_pred,
    average="weighted"
)


f1_value = f1_score(
    y_true,
    y_pred,
    average="weighted"
)



print("Accuracy :", accuracy_value)

print("Precision :", precision_value)

print("Recall :", recall_value)

print("F1 Score :", f1_value)

# ==================3
print(
classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_
)
)
# =============================================================== BERT Sentiment Confusion Matrix figure 
import matplotlib.pyplot as plt
import seaborn as sns


cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(figsize=(6,5))


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "BERT Sentiment Confusion Matrix"
)


plt.show()

# ==============================================================  BERT Model Performance  ##
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


values = [
    accuracy_value,
    precision_value,
    recall_value,
    f1_value
]


plt.figure(figsize=(7,5))


plt.bar(
    metrics,
    values
)


plt.ylim(
    0,
    1
)


plt.ylabel(
    "Score"
)


plt.title(
    "BERT Model Performance"
)


plt.show()

# ==============================================================
history = trainer.state.log_history


train_loss=[]

eval_loss=[]


for item in history:

    if "loss" in item:

        train_loss.append(
            item["loss"]
        )


    if "eval_loss" in item:

        eval_loss.append(
            item["eval_loss"]
        )
        
# ====================================================== Training Loss Curve
plt.figure(figsize=(8,5))

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.xlabel(
    "Steps"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training Loss Curve"
)


plt.legend()

plt.grid()

plt.show()
# =====================================  Validation Loss Curve
plt.figure(figsize=(8,5))


plt.plot(
    eval_loss,
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Validation Loss Curve"
)


plt.legend()

plt.grid()

plt.show()



# =====================================  PART __3  Model Evaluation and Visualization  ######

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ====================================================              # Predict test data

prediction_output = trainer.predict(test_dataset)


# Extract predicted labels

y_pred = np.argmax(
    prediction_output.predictions,
    axis=1
)


# True labels

y_true = prediction_output.label_ids


print("Prediction Completed")
# ============================ ========================  3.1              Calculate Accuracy
accuracy_value = accuracy_score(
    y_true,
    y_pred
)

print(
    "Accuracy:",
    accuracy_value
)

# =============================================== 3.2            Calculate Precision
precision_value = precision_score(
    y_true,
    y_pred,
    average="weighted"
)


print(
    "Precision:",
    precision_value
)
# ================================================ 3.3               Calculate Recall
recall_value = recall_score(
    y_true,
    y_pred,
    average="weighted"
)


print(
    "Recall:",
    recall_value
)
# =============================== =========================  3.4           Calculate F1 Score
f1_value = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print(
    "F1 Score:",
    f1_value
)

# ====================================================   3.5 Complete Classification Report
print(
    classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )
)

# ==================================================== 3.6          Confusion Matrix
cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(figsize=(7,5))


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)


plt.xlabel("Predicted Sentiment")

plt.ylabel("Actual Sentiment")

plt.title(
    "BERT Sentiment Analysis Confusion Matrix"
)


plt.show()
# ======================================================3.7    plot Performance comparison graph
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

 
scores = [
    accuracy_value,
    precision_value,
    recall_value,
    f1_value
]


plt.figure(figsize=(8,5))


plt.bar(
    metrics,
    scores
)


plt.ylim(0,1)


plt.ylabel(
    "Score"
)


plt.title(
    "BERT Model Performance"
)


plt.show()

# ================         =================================== 3.8            Training and Validation Loss Graph
history = trainer.state.log_history


train_loss = []

eval_loss = []


for item in history:

    if "loss" in item:

        train_loss.append(
            item["loss"]
        )


    if "eval_loss" in item:

        eval_loss.append(
            item["eval_loss"]
        )


print("Training loss values:", len(train_loss))

print("Validation loss values:", len(eval_loss))

# ======================================================== 3.9          Training Loss Plot
plt.figure(figsize=(8,5))


plt.plot(
    train_loss,
    marker="o"
)


plt.xlabel(
    "Training Steps"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training Loss Curve"
)


plt.grid()


plt.show()
# ============== =======================================     3.10                Validation Loss Plot
if len(eval_loss)>0:

    plt.figure(figsize=(8,5))


    plt.plot(
        eval_loss,
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Validation Loss Curve"
    )


    plt.grid()


    plt.show()

else:

    print("Validation loss not available")

# =========================== =================================== 3.11             Test Model with New Sentences
texts = [

    "This product is amazing and I love it",

    "The service was very bad",

    "The movie was average"

]


inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model.to(device)


inputs = {
    key:value.to(device)
    for key,value in inputs.items()
}


model.eval()


with torch.no_grad():

    outputs = model(**inputs)



prediction = torch.argmax(
    outputs.logits,
    dim=1
)



sentiments = encoder.inverse_transform(
    prediction.cpu().numpy()
)



for text, sentiment in zip(texts, sentiments):

    print("------------------------")

    print("Review:")
    print(text)

    print("Predicted Sentiment:")
    print(sentiment)