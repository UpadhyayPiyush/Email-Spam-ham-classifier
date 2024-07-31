
import numpy as np
import pandas as pd


 df=pd.read_csv(r"spam.csv",encoding='latin1')





df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace =True)   #drop last 3 column.... it has lots of null values


#renaming...

df.rename(columns={'v1':'target','v2':'text'},inplace=True)


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


encoder.fit_transform(df['target'])    #encoding text to 0 and 1


df['target']=encoder.fit_transform(df['target'])   


df.drop_duplicates(keep='first')  #removing duplicates from dataset...


df=df.drop_duplicates(keep='first')




import nltk


nltk.download('punkt')


df['num_characters']=df['text'].apply(len)  #show number of characters in each sms by counting


df['text'].apply(lambda x:nltk.word_tokenize(x))   ###this break the whole text into a words....


df['text'].apply(lambda x:len(nltk.word_tokenize(x)))    #here we use len() to counting the number of words in each text...


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))  #creating another new column and store in it...


df['text'].apply(lambda x:nltk.sent_tokenize(x))       #breaking into sentences....


df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))     #counting number of sentences..


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))   #creating new column and store in it...


df[['num_characters','num_words','num_sentences']].describe()     #apply describe function on new creating columns only


#for ham messages only...

df[df['target']==0][['num_characters','num_words','num_sentences']].describe()    


#for spam messages...

df[df['target']==1][['num_characters','num_words','num_sentences']].describe() 


import seaborn as sns



#First step ::::: Lower case....

def transform_text(text):
    text=text.lower()
    return text


transform_text('HEY Whatsapp')


#Second step :::: Tokenization(word)

def transform_text(text):
    text=nltk.word_tokenize(text)
    return text


transform_text("HEY Whatsapp")


#Third step ::::: Removing special characters...

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    return y





transform_text("HEY Whatsapp You got 80 %% in your exam")   #hence it removed special character....


nltk.download('stopwords')


from nltk.corpus import stopwords
stopwords.words('english')


import string 
string.punctuation


#Fourth step ::: Removing stopwords and punctuation..

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    
            
    return y





transform_text("HEY Whatsapp You got 80 %% in your exam") 


#Fifth step ::: Stemming....

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('dancing')


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
        
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)        



transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...') 


df['text'][0]


df['text'].apply(transform_text)


df['transformed_text']=df['text'].apply(transform_text)


pip install WordCloud


from wordcloud import WordCloud     # word cloud is a visual representation of text data where the size of each word indicates its frequency or importance in the text. 
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))     #generate function is used to generate a word cloud from a text corpus. 


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" ")) 


spam_corpus=[]

for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


len(spam_corpus)


# from collections import Counter
# sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()


###Error...... show most common 30 words for spam or not spam



from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


X=tfidf.fit_transform(df['transformed_text']).toarray()


y=df['target'].values     #extracting target column into y variable....


from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)     #why we use random state =2?


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


#tfidf -> mnb


###Machine learning models...... Almost all classification algos...

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


pip install xgboost


#making an object for every model...

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


#making a dictionary...in which(keys=algos name,value=object name)

clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


#simple making a function as a train_classifer for calculating accuracy and precision

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


train_classifier(svc,X_train,y_train,X_test,y_test)     #find acc and prec for svc


#its gives the list of acc and precision scores for all models...


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


performance_df


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


performance_df1 


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


#checking accuracy and precision after using max_features=3000 in tfidf

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


#checking accuracy and precision after scaling 

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


new_df = performance_df.merge(temp_df,on='Algorithm')


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


#checking for num_chars also...

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


new_df_scaled.merge(temp_df,on='Algorithm')


# Voting Classifier...A voting classifier is a machine learning model that combines the predictions of multiple other models (called base classifiers) to make a final prediction. The idea is to leverage the strengths of different models to improve overall performance.
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


voting.fit(X_train,y_train)


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# Applying stacking also known as stacked generalization, is an ensemble learning technique that combines multiple base models to improve prediction accuracy.
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


from sklearn.ensemble import StackingClassifier


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))