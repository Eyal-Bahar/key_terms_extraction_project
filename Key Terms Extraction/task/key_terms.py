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


def get_most_common(token_list):
    most_common = []
    joined_tokens = []
    for tokens in token_list:
        joined_tokens = " ".join(tokens)
        # all_joined_tokens.append(" ".join(tokens))

        tf_idf_scores, terms = tfidf_matrix([joined_tokens])
        tf_idf_scores = tf_idf_scores.toarray()
        word_rate = []
        for row, text_idx in enumerate(range(tf_idf_scores.shape[0])):
            for col, tfidf_score in enumerate(tf_idf_scores[text_idx]):
                if tfidf_score != 0:
                    word_rate.append((terms[col], tfidf_score))
            sorted_words = sorted(word_rate, key=lambda x: (x[1],x[0]), reverse=True)[0:5]
            most_common.append([word[0] for word in sorted_words])
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
