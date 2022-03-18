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
# nltk.download('averaged_perceptron_tagger')


def get_rood_from_xml(xml_path=None):
    if xml_path is None:
        xml_path = r"C:\hyper_skill\Key Terms Extraction\Key Terms Extraction\task\news.xml"
    return etree.parse(xml_path).getroot()


def get_most_common(token):
    token = sorted(token, reverse=True)
    wat = Counter(token).most_common(5)
    most_common = [key for key, value in wat]
    return most_common


def filter_token(token):
    filter_list = stopwords.words('english')
    filter_list += list(string.punctuation)
    filtered_token = [inst for inst in token if inst not in filter_list]
    return filtered_token


def extract_nouns(filtered_tokens):
    nouned_tokens = []
    for token in filtered_tokens:
        tagged = nltk.pos_tag([token])
        if tagged[0][1] == "NN":
            nouned_tokens.append(tagged[0][0])
    return nouned_tokens

def main():

    root = get_rood_from_xml()

    lemmatizer = WordNetLemmatizer()
    for corpus in root[0]:
        for value in corpus:
            if value.get('name') == "head":
                print(value.text + ":")
            if value.get('name') == "text":
                raw_tokens = word_tokenize(value.text.lower())
                lemmatized_tokens = [lemmatizer.lemmatize(raw_token) for raw_token in raw_tokens]
                filtered_tokens = filter_token(lemmatized_tokens)
                noun_tokens = extract_nouns(filtered_tokens)
                most_common = get_most_common(noun_tokens)
        print(*most_common)

if __name__ == '__main__':
    main()