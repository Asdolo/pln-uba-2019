from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
    'mnb': MultinomialNB
}


def feature_dict(sent, i):
    """Feature dictionary for a given sentence and position.

    sent -- the sentence.
    i -- the position.
    """
    sentLength = len(sent)
    word = sent[i]

    if i == sentLength - 1:
        nextWord = '</s>'
    else:
        nextWord = sent[i+1]

    previousWord = '<s>' if i == 0 else sent[i-1]

    return {
            'word_lowercase': word.lower(),
            'is_uppercase': word.isupper(),
            'is_capitalized': word.istitle(),
            'is_digit': word.isdigit(),
            'previous_word_lowercase': previousWord.lower(),
            'previous_word_is_uppercase': previousWord.isupper(),
            'previous_word_is_capitalized': previousWord.istitle(),
            'previous_word_is_digit': previousWord.isdigit(),
            'next_word_lowercase': nextWord.lower(),
            'next_word_is_uppercase': nextWord.isupper(),
            'next_word_is_capitalized': nextWord.istitle(),
            'next_word_is_digit': nextWord.isdigit(),
        }

def get_sents_without_tags(tagged_sents):
        sents = []
        for tagged_sent in tagged_sents:
            sent = []
            for word, _ in tagged_sent:
                sent.append(word)
            sents.append(sent)

        return sents

def get_features_of_tagged_sents(tagged_sents):
    sents_without_tags = get_sents_without_tags(tagged_sents)
    sents_with_features = []
    y_true = []
    for sent_index, tagged_sent in enumerate(tagged_sents):
        for word_index, tuple in enumerate(tagged_sent):
            word, tag = tuple
            x = feature_dict(sents_without_tags[sent_index], word_index)
            sents_with_features.append(x)
            y_true.append(tag)

    return sents_with_features, y_true

class ClassifierTagger:
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr'):
        """
        clf -- classifying model, one of 'svm', 'lr', 'mnb' (default: 'lr').
        """
        self.pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('clf', classifiers[clf]())
        ])

        self.knownWords = []

        self.fit(list(tagged_sents))

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        for taggedSent in tagged_sents:
            for word, _ in taggedSent:
                if not word in self.knownWords:
                    self.knownWords.append(word)

        X, y_true = get_features_of_tagged_sents(tagged_sents)

        self.pipeline.fit(X, y_true);

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        return [self.tag(sent) for sent in sents]

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """

        X = [feature_dict(sent, i) for i, word in enumerate(sent)]
        y_pred = self.pipeline.predict(X)
        return y_pred

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.knownWords
