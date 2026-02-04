import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def clean_text(txt):
    txt = ''.join([i for i in txt if i.isascii()])
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = ''.join([i for i in txt if not i.isdigit()])
    words = word_tokenize(txt)
    return " ".join([w for w in words if w not in stop_words])
