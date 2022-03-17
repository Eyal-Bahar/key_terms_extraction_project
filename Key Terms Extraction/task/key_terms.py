# Write your code here
from nltk import ngrams, FreqDist
from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter, OrderedDict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk
# nltk.download('wordnet')

xml_file = r"C:\hyper_skill\Key Terms Extraction\Key Terms Extraction\task\news.xml"
root = etree.parse(xml_file).getroot()
# etree.dump(root)
tokens = []
lemmatizer = WordNetLemmatizer()
for corpus in root[0]:
    for value in corpus:
        if value.get('name') == "head":
            print(value.text + ":")
        if value.get('name') == "text":
            raw_token = word_tokenize(value.text.lower())
            lemma_token = [lemmatizer.lemmatize(inst) for inst in raw_token]
            filter_list = stopwords.words('english')
            filter_list += list(string.punctuation)
            token = [inst for inst in lemma_token if inst not in filter_list]
            token = sorted(token, reverse=True)
            wat = Counter(token).most_common(5)
            most_common = [key for key, value in wat]
    print(*most_common)