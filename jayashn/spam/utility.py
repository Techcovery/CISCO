from pandas import DataFrame
import os
import numpy

SKIP_FILES = {'cmds'}
NEWLINE = '\n'

HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('enron1/spam',   SPAM),
    ('enron1/ham',    HAM),
    ('enron2/spam',   SPAM),
    ('enron2/ham',    HAM), 
]

def read_files(path):
    lines = []
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    f = open(file_path, encoding="latin-1")
                    content = f.read()
                    f.close()
                    yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

print (data)
data.to_csv("my_email.csv")
