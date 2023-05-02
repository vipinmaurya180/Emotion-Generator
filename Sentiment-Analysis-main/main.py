import string
from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
import matplotlib.pyplot as plt
import nltk
nltk.download("vader_lexicon")

## ** reading the csv file**
df = pd.read_csv('test.csv')        

## ** selecting the particular cell **
cell = df.loc[15,"text"]              


texts =  cell.translate(str.maketrans('','',string.punctuation))


## ** making the text into tokenize form  and removing the punctuations **
#token = RegexpTokenizer(r'\w+')
tokenized_words = word_tokenize(texts)     


## *** collecting the stop words**
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


## *** Removing the stop words ***
tokenize_tokenized_words_without_stopwords = []
for word in tokenized_words:
    if word not in stop_words:
        tokenize_tokenized_words_without_stopwords.append(word)

#print(tokenize_tokenized_words_without_stopwords)

# ***emotion forming ***
emotion_list = []
with open('emotion.txt','r')  as file:
    for line in file:
        clear_line =  line.replace('\n','').replace(',','').replace("'", '').strip()
        word,emotion = clear_line.split(':')

        if word in tokenize_tokenized_words_without_stopwords:
            emotion_list.append(emotion)

print('Emotion list- ', emotion_list)

w = Counter(emotion_list)
print(w)


## find the sentiment of the statement **
from nltk.sentiment import SentimentIntensityAnalyzer
sent =  SentimentIntensityAnalyzer()
senti_dict = sent.polarity_scores(texts)
print('Sentiment list- ',sent.polarity_scores(texts))
print(senti_dict['neg'])

'''if (senti_dict['neg']> senti_dict['neu'] )and (senti_dict['neg']>senti_dict['pos']):
    print('Negative')
elif (senti_dict['pos']>senti_dict['neu'] )and (senti_dict['pos'] < senti_dict['neg']):
    print('Positive')
else:
    print('Neutral')'''

if (senti_dict['neg']> senti_dict['neu'] )and (senti_dict['neg']>senti_dict['pos']):
    print('Negative')
else:
    print('Positive')

## ** counting the frequency **

from nltk.probability import FreqDist
print(FreqDist(tokenize_tokenized_words_without_stopwords))
fd = FreqDist(tokenize_tokenized_words_without_stopwords)

print(fd.most_common(7))
fd.plot(50,cumulative=False)
fig,axl = plt.subplots()
axl.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.show()

