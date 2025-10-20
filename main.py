#Import Libraries
import torch
import streamlit as st
import pandas as pd
from sklearn.utils import shuffle 
import tldextract as t
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalMaxPooling1D, Conv1D, GRU, BatchNormalization
from transformers import TFDistilBertModel, DistilBertTokenizer
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import custom_object_scope
%matplotlib inline

f_news1 = pd.read_csv('/Users/taahashaikh/Downloads/gossipcop_fake.csv')
r_news1 = pd.read_csv('/Users/taahashaikh/Downloads/gossipcop_real.csv')
p_fnews2 = pd.read_csv('/Users/taahashaikh/Downloads/politifact_fake.csv')
p_rnews2 = pd.read_csv('/Users/taahashaikh/Downloads/politifact_real.csv')

f_news1.head()
f_news1.isna().sum()
f_news1.info()

r_news1.head()
r_news1.info()

p_fnews2.head()
p_fnews2.info()

p_rnews2.head()
p_rnews2.info()

# Add Labels fake news is 0 and real news is 1
f_news1['news_status'] = 0
r_news1['news_status'] = 1
p_fnews2['news_status'] = 0
p_rnews2['news_status'] = 1

data = pd.concat([f_news1, r_news1, p_fnews2, p_rnews2], ignore_index=True)
data1 = shuffle(data).reset_index(drop=True)
data1.head()

data1.info()

def extract_url(url):
    try:
        parsed = t.extract(url)
        parsed = '.'.join([i for i in [parsed.subdomain, parsed.domain, parsed.suffix] if i])
        return parsed
    except:
        return 'Na'


data1.loc[:, 'source_domain'] = data1.loc[:, 'news_url'].apply(lambda x: extract_url(x))
data1.head()

def get_count(txt):
    if type(txt) is not str:
        return 0
    else:
        x = txt.split('\t')
        return len(x)


data1.loc[:, 'tweet_num'] = data1.loc[:,'tweet_ids'].apply(lambda x: get_count(x))
data1.head()

#data1 = data1.drop(columns=['id', 'tweet_ids'])
data1 = data1[['title', 'news_url', 'source_domain', 'tweet_num', 'news_status']]
data1.head()

#data_set = data1.to_csv('FakeNewsDataSet.csv', index=False)
dataset = pd.read_csv('/Users/taahashaikh/project/FakeNewsDataset.csv')
dataset

x = dataset['title'].astype(str)
y = dataset['news_status']
y

encoder = LabelEncoder()
y = encoder.fit_transform(y)

xtrain, xtest, ytrain, ytest = train_test_split(x, 
                                                y, 
                                                test_size=0.2, 
                                                random_state=42, 
                                                stratify=y)

xtrain
xtest
ytrain
ytest

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

def extract_text(texts, tokenizer, max_len=72):
    return tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')

train_enc = extract_text(xtrain, tokenizer, 72)
test_enc = extract_text(xtest, tokenizer, 72)

#tf.keras.backend.clear_session()
print(train_enc['input_ids'].shape)
print(train_enc['attention_mask'].shape)

input_ids = Input(shape=(72,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(72,), dtype=tf.int32, name='attention_mask')

bert_ouput = bert_model(input_ids, attention_mask=attention_mask)
x = bert_ouput.last_hidden_state

x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)

x = BatchNormalization()(x)

x = tf.expand_dims(x, -1)
x = GRU(64, dropout=0.3, recurrent_dropout=0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=Adam(learning_rate=2e-5), loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
            ModelCheckpoint('best_hybrid_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-6, verbose=1)]

model_fit = model.fit({'input_ids': train_enc['input_ids'], 'attention_mask': train_enc['attention_mask']},
                      ytrain,
                      validation_data=({'input_ids': test_enc['input_ids'], 'attention_mask': test_enc['attention_mask']},
                                       ytest),
                      epochs=5, batch_size=16, callbacks=callbacks)

with custom_object_scope({'TFDistilBertModel': TFDistilBertModel}):
    loaded_model = tf.keras.models.load_model("best_hybrid_model.h5", compile=False)

tokenizer.save_pretrained('tokenizer/')

loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                     loss='binary_crossentropy',
                     metrics=['accuracy']
                    )

loss, acc = loaded_model.evaluate({'input_ids': test_enc['input_ids'], 'attention_mask': test_enc['attention_mask']}, ytest)
print(f"✅ Best Model Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

y_pred_prob = loaded_model.predict(
    {'input_ids': test_enc['input_ids'], 'attention_mask': test_enc['attention_mask']}
)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("📊 Classification Report:")
print(classification_report(ytest, y_pred, target_names=["Fake", "Real"]))
#def predict_news(text):
    #enc = extract_text([text], tokenizer)
    #pred = model.predict({'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']})[0][0]
    #return '🟩 Real News' if pred > 0.5 else '🟥 Fake News'

#print(predict_news("Pakistan, Afghanistan agree to 48-hour ceasefire: Pakistani Government"))

plt.figure(figsize=(6,4))
plt.plot(model_fit.history['accuracy'], label='Train Accuracy', color='limegreen')
plt.plot(model_fit.history['val_accuracy'], label='Val Accuracy', color='gold')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

