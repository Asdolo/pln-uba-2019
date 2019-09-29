"""Train a sequence tagger.

Usage:
  train.py [options] -c <path> -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: badbase]:
                  badbase: Bad baseline
                  base: Baseline
                  lr-classifier: LogisticRegression Classifier
                  svm-classifier: LinearSVC Classifier
                  mnb-classifier: MultinomialNB Classifier
  -c <path>     Ancora corpus path.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger, BadBaselineTagger
from tagging.classifier import ClassifierTagger


models = {
    'badbase': BadBaselineTagger,
    'base': BaselineTagger,
    'lr-classifier': ClassifierTagger,
    'svm-classifier': ClassifierTagger,
    'mnb-classifier': ClassifierTagger,
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader(opts['-c'], files)
    sents = corpus.tagged_sents()

    # train the model
    model_arg = opts['-m']
    model_class = models[model_arg]
    model = model_class(sents, model_arg.split('-')[0]) if model_arg.endswith('classifier') else model_class(sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
