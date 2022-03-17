# Write your code here
from nltk import ngrams, FreqDist
from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter, OrderedDict

xml_file = r"C:\hyper_skill\Key Terms Extraction\Key Terms Extraction\task\news.xml"
root = etree.parse(xml_file).getroot()
# etree.dump(root)
tokens = []
for corpus in root[0]:
    for value in corpus:
        if value.get('name') == "head":
            print(value.text + ":")
        if value.get('name') == "text":
            token = word_tokenize(value.text.lower())
            token = sorted(token, reverse=True)
            wat = Counter(token).most_common(5)
            most_common = [key for key, value in wat]
    print(*most_common)