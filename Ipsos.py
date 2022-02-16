

import re
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import csv
import time
import torch
import random
import torchtext
import pandas as pd
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True


class Data():

    def __init__(self, path) -> None:
        self.df = pd.read_excel(path, engine='openpyxl')


    def get_shape(self):
        return self.df.shape


    def get_info(self):
        return self.df.info()


    def get_df(self):
        return self.df


    def rename_columns(self):

        if 'ID' not in self.df.columns :
            self.df.rename(columns={'SIA_ID': 'ID', 'Sound Bite Text': 'text', 'Source Type' : 'source', 'Published Date (GMT-05:00) New York' : 'time'}, inplace=True)
        else : 
            return "[!] Columns are aleady renamed ! "


    def annotate_data(self, output_path):

        print('[!] Annotating data ...')

        if 'ID' not in self.df.columns : 
            self.rename_columns()

        self.df = self.df.iloc[:, :-1]
        print('[!] Cleaning text ...')

        self.df['cleaned_text'] = self.df.text.apply(self.clean_text)
        self.get_polarity()

        print('[!] Done :) ')
        self.df.to_csv(output_path, index = False)
        print(f'[!] Data saved to {output_path} ! ')

       
    def clean_text(self, text):  
        # Removing tags
        tags = r'@[^ ]+'   
        # Removnig urls                
        rm_urls = r'https?://[A-Za-z0-9./]+'  
        pat3 = r'\'s'   
        # Removing Hashes                   
        rm_hash = r'\#\w+'                     
        pat5 = r'&amp '                     
        pat6 = r'[^A-Za-z\s]'               
        combined_pat = r'|'.join((tags, rm_urls,pat3,rm_hash,pat5, pat6))
        text = re.sub(combined_pat,"",text).lower()
        return text.strip()


    def get_polarity(self):
        # To annotate data, I used Textblob
        # That it returns polarity of each sentence
        print("[!] Adding Polarity to Dataframe ...")
        for row in self.df.itertuples():
                tweet = self.df.at[row[0], 'cleaned_text']
                # run sentiment using TextBlob
                analysis = TextBlob(tweet)
                # If polarity > 0 -> That means it's a positive sentence
                if analysis.sentiment[0]>0:
                    self.df.at[row[0], 'Sentiment'] = "Positive"
                elif analysis.sentiment[0]<0:
                    self.df.at[row[0], 'Sentiment'] = "Negative"
                else:
                    self.df.at[row[0], 'Sentiment'] = "Neutral"

        print('[!] Removing Neutral ...')
        self.df = self.df[self.df.Sentiment != "Neutral"]
        self.df = self.df[['text', 'source', 'Sentiment']]
         


class Viz():

    def __init__(self, path) -> None:

        self.df = pd.read_csv(path)


    def hist_source(self):

        plt.figure(figsize = (14,8))
        ax = sns.countplot(x="source", data= self.df)
        plt.title('Different sources of our data')
        plt.ylabel('Count')
        plt.xlabel('Sources')
        plt.show()
        # return ax 


    def pie_source(self):

        plt.figure(figsize = (14,14))
        labels = self.df.source.unique()
        counts = [self.df.source[self.df.source == el].count() for el in labels]
        #define Seaborn color palette to use
        colors = sns.color_palette('Set2') # [0:5] # Paired
        #create pie chart
        plt.pie(x= counts, autopct="%.1f%%", explode=[0.8]*9, labels=labels, pctdistance=0.5, colors = colors)
        plt.show()


    def hist_target(self):

        plt.figure(figsize = (8,8))
        ax = sns.countplot(x="Sentiment", data= self.df)
        plt.title('Distribution of Sentiment')
        plt.ylabel('Count')
        plt.xlabel('Sentiments')
        plt.show()
        # return ax


    def hist_target_source(self):

        plt.figure(figsize = (14, 8))
        ax = sns.countplot(x="Sentiment", hue="source", data= self.df)
        plt.show()
        # return ax 


    def plot_wordcloud(self, type = 'default'):

        if type == 'default' :
            data_neg = self.df.text
        elif type == 'Negative' :
            data_neg = self.df.text[self.df.Sentiment == 'Negative']
        elif type == 'Positive' :
            data_neg = self.df.text[self.df.Sentiment == 'Positive']
        else : 
            return "Error ! For type, choose : Negative, Positive or default  ! "

        plt.figure(figsize = (20,20))
        wc = WordCloud(background_color='white', max_words = 1000 , width = 1600 , height = 800,
                    collocations=False).generate(" ".join(data_neg))
        plt.axis("off")
        plt.imshow(wc)




class ML():

    # global model 

    def __init__(self, path) :
        self.df = pd.read_csv(path)
        # Whole_data is the one that I'm going to project my models on it
        self.whole_data = pd.read_excel('Data_internship_test_anonymized.xlsx', engine='openpyxl')
        self.porter = PorterStemmer()


    def get_whole_data(self):
        return self.whole_data


    def pre_process(self):

        print('[!] Renaming columns for whole data ...')
        self.whole_data.rename(columns={'Original_verbatim': 'text'}, inplace = True)
        
        print('[!] Removing Stop Words ...')
        self.df['text'] = self.df.text.apply(self.remove_stop_words)
        self.whole_data.text = self.whole_data.text.apply(self.remove_stop_words)

        print('[!] Cleaning texts >> [ Removing punctuations, urls, tags ...] ...')
        self.df['text'] = self.df.text.apply(self.clean_text)
        self.whole_data.text = self.whole_data.text.apply(self.clean_text)

        print('[!] Applying Stemming ...')
        self.df['text'] = self.df['text'].apply(self.stemSentence)
        self.whole_data.text = self.whole_data.text.apply(self.stemSentence)

        print('[!] Splitting data ...')
        X = self.df.text
        y = self.df.Sentiment
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 666)

        print('[!] Applying TfidfVectorizer ...')
        vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
        vectoriser.fit(X_train)
        X_train = vectoriser.transform(X_train)
        X_test  = vectoriser.transform(X_test)
        whole_X = vectoriser.transform(self.whole_data.text)

        return X_train, X_test, y_train, y_test, whole_X


    def fit_model(self, model_name, X_train, y_train):

        print('[!] Fitting the model ... ')
        if model_name == 'SVM' :
            model = SVC()

        elif model_name == 'BNB' :
            model = BernoulliNB()

        elif model_name == 'LR' :
            model = LogisticRegression()

        else : 
            return "[!] Error ! Choose : \n- SVM for  SVM model\n- BNB for BernoulliNB model\nLR for LogisticRegression  model"

        model.fit(X_train, y_train)

        return model 


    def predict_evaluate(self, model, X_test, y_test): # X_train, y_train, 

        
        # model = self.fit_model(model_name, X_train, y_train)
        print("[!] Predicting values ...")
        # Predict values for Test dataset
        y_pred = model.predict(X_test)

        print("[!] Evaluating the model ...")
        # Print the evaluation metrics for the dataset.
        print(classification_report(y_test, y_pred))

        # Compute and plot the Confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)

        categories = ['Negative','Positive']
        group_names = ['True Neg','False Pos', 'False Neg','True Pos']
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        # Plotting confusion matrix
        sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
        xticklabels = categories, yticklabels = categories)
        plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
        plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
        plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20) 

        return y_pred   


    def remove_stop_words(self, text):
        # Removing stop words like : the, is ... 
        # I used nltk. Don't forget to download nltk.download('stopwords')
        # It's not included in nltk
        stop_words = set(stopwords.words('english'))

        word_tokens = word_tokenize(text)

        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
                
        return " ".join(filtered_sentence)


    def clean_text(self, text):  
        # Cleaning texts same as the first class 
        tags = r'@[^ ]+'                   
        rm_urls = r'https?://[A-Za-z0-9./]+'  
        pat3 = r'\'s'                      
        rm_hash = r'\#\w+'                     
        pat5 = r'&amp '                     
        pat6 = r'[^A-Za-z\s]'               
        combined_pat = r'|'.join((tags, rm_urls,pat3,rm_hash,pat5, pat6))
        text = re.sub(combined_pat,"",text).lower()
        return text.strip()


    def stemSentence(self, sentence):
        # To apply stemmer on each sentence
        # I used PorterStemmer class
        token_words = word_tokenize(sentence)
        token_words
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(self.porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)




# Building LSTM model 
class RNN(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)        
        
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        

    def forward(self, text):
        # text dim: [sentence length, batch size]
        
        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]
        
        output = self.fc(hidden)
        return output


class DL():

    def __init__(self, path) -> None:
        if 'csv' in path :
            self.df = pd.read_csv(path)[['text', 'Sentiment']]
        else : 
            self.df = pd.read_excel(path, engine='openpyxl')['Original_verbatim']
    

    def pre_processing(self):

        print('[!] Cleaning data ...')
        self.df.text = self.df.text.apply(self.clean_text)

        print('[!] Splitting data ...')
        df_train, df_test = train_test_split(self.df, test_size=0.15, random_state = 666)
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state = 666)
        
        print('[!] Saving dataset ...')
        df_train.to_csv('DATASET/train_data.csv', index = False)
        df_test.to_csv('DATASET/test_data.csv', index = False)
        df_val.to_csv('DATASET/val_data.csv', index = False)

        print('[!] Defining the feature processing ...')
        ### Defining the feature processing
        TEXT = torchtext.legacy.data.Field(
            tokenize='spacy', # default splits on whitespace
            tokenizer_language='en_core_web_sm'
        )
        print('[!] Defining the sentiment processing ...')
        ### Defining the SENTIMENT processing
        SENTIMENT = torchtext.legacy.data.LabelField(dtype=torch.long)

        fields = [('TEXT_COLUMN_NAME', TEXT), ('LABEL_COLUMN_NAME', SENTIMENT)]

        train_data = torchtext.legacy.data.TabularDataset(
            path= 'DATASET/train_data.csv', format='csv',
            skip_header=True, fields=fields)

        test_data = torchtext.legacy.data.TabularDataset(
            path= 'DATASET/test_data.csv', format='csv',
            skip_header=True, fields=fields)

        val_data = torchtext.legacy.data.TabularDataset(
            path= 'DATASET/val_data.csv', format='csv',
            skip_header=True, fields=fields)

        print('[!] Word -> id_word')
        TEXT.build_vocab(train_data, max_size= 20000)
        SENTIMENT.build_vocab(train_data)

        print(f'\t -> Vocabulary size: {len(TEXT.vocab)}')
        print(f'\t -> Number of classes: {len(SENTIMENT.vocab)}')

        print(f'[!] Target data : {SENTIMENT.vocab.stoi}')

        return train_data, test_data, val_data


    def clean_text(self, text):  

        tags = r'@[^ ]+'                   
        rm_urls = r'https?://[A-Za-z0-9./]+'  
        pat3 = r'\'s'                      
        rm_hash = r'\#\w+'                     
        pat5 = r'&amp '                     
        pat6 = r'[^A-Za-z\s]'               
        combined_pat = r'|'.join((tags, rm_urls,pat3,rm_hash,pat5, pat6))
        text = re.sub(combined_pat,"",text).lower()
        return text.strip()



