from collections import defaultdict
import operator


def cero():
    return 0


def dd():
    return defaultdict(cero)


class BadBaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for all words.
        """
        self._default_tag = default_tag

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        return self._default_tag

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return True


class BaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for unknown words.
        """
        self.defaultTag = default_tag

        self.wordTagsCount = defaultdict(dd)
        for taggedSent in tagged_sents:
            for word, tag in taggedSent:
                self.wordTagsCount[word][tag] += 1

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self.defaultTag

        wordTagsDict = self.wordTagsCount[w]
        return max(wordTagsDict.items(), key=operator.itemgetter(1))[0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return len(self.wordTagsCount[w]) == 0
