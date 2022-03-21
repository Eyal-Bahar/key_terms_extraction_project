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
        xml_path = r"C:\hyper_skill\Key Terms Extraction\Key Terms Extraction\task\news4.xml"
    return etree.parse(xml_path).getroot()


def tfidf_matrix(dataset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True,
                                 analyzer='word', ngram_range=(1, 1),
                                 stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(dataset)
    terms = vectorizer.get_feature_names_out()
    return tfidf_matrix, terms


def build_all_joined_tokens(token_list):
    all_joined_tokens = []
    for tokens in token_list:
        joined_tokens = " ".join(tokens)
        all_joined_tokens.append(joined_tokens)
    return all_joined_tokens


def most_common_counter(tf_idf_scores, terms):
    most_common = []
    row = 0
    dimension = tf_idf_scores.shape
    while row < dimension[0]:
        column = 0
        word_rate = []
        while column < dimension[1]:
            if tf_idf_scores[row][column] != 0:
                word_rate.append((terms[column], tf_idf_scores[row][column]))  # first value = word , second value = rate
            column += 1
    #    sorting word
        sorted_words = sorted(word_rate, key=lambda x: (x[1], x[0]), reverse=True)[0:5]
        most_common.append([word[0] for word in sorted_words])
        row = row + 1
    return most_common


def get_most_common(token_list):
    all_joined_tokens = build_all_joined_tokens(token_list)

    tf_idf_scores, terms = tfidf_matrix(all_joined_tokens)
    tf_idf_scores = tf_idf_scores.toarray()

    most_common = most_common_counter(tf_idf_scores, terms)

    return most_common


def filter_token(token):
    filter_list = stopwords.words('english')
    filter_list += list(string.punctuation)
    filter_list +=  ["ha", "wa", "u", "a"]
    filtered_token = [inst for inst in token if inst not in filter_list]
    return filtered_token


def extract_nouns(filtered_tokens):
    nouned_tokens = []
    for token in filtered_tokens:
        tagged = nltk.pos_tag([token])
        if tagged[0][1] == "NN":
            nouned_tokens.append(tagged[0][0])
    return nouned_tokens


def extract_tokens_and_headlines(root):
    lemmatizer = WordNetLemmatizer()
    token_list = []
    head_line_list = []
    for corpus in root[0]:
        for value in corpus:
            if value.get('name') == "head":
                head_line_list.append(value.text + ":")
            if value.get('name') == "text":
                raw_tokens = word_tokenize(value.text.lower())
                lemmatized_tokens = [lemmatizer.lemmatize(raw_token) for raw_token in raw_tokens]
                filtered_tokens = filter_token(lemmatized_tokens)
                noun_tokens = extract_nouns(filtered_tokens)
                token_list.append(noun_tokens) # In stage 3/4: most_common = get_most_common(noun_tokens)
    return head_line_list, token_list




def main():
    root = get_rood_from_xml()
    headlines, token_list = extract_tokens_and_headlines(root)
    most_common = get_most_common(token_list)
    for i, headline in enumerate(headlines):
        print(headline)
        print(*most_common[i])

if __name__ == '__main__':
    main()
