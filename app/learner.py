import json
import os
import random
import numpy as np
from app import settings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


class Learner():

    def __init__(self, dataDir):
        self.docs = self.loadItems(dataDir)
        self.doc_ids = [doc["id"] for doc in self.docs]
        self.doc_index = {docId:index for index, docId in enumerate(self.doc_ids)}
        print("Loaded %d docs" % len(self.docs))
        print(list(self.doc_index.items())[:10])

        self.vectorizer = TfidfVectorizer(stop_words='english',
                                     ngram_range=settings.NGRAM_RANGE,
                                     token_pattern="[a-zA-Z]{2,}",
                                     min_df=settings.MIN_DOC_FREQUENCY,
                                     max_df=settings.MAX_DOC_FREQUENCY)

        self.vectors = self.vectorizer.fit_transform([" ".join([doc["title"], doc["description"]]) for doc in self.docs])
        self.sorted_vocab = sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1])

        print("Extracted %d features with a minimum frequency of %s across %d documents." %
              (self.vectors.shape[1], settings.MIN_DOC_FREQUENCY, self.vectors.shape[0]))
        self.model = SGDClassifier(loss="log")
        self.alreadySeen = set()
        # TODO: get batches

    def getCandidates(self):
        candidates = []
        if not self.alreadySeen:
            print("--> Getting random candidates")
            return random.sample(self.docs, settings.BATCH_SIZE)

        self.print_features(10)
        scores = self.model.predict_proba(self.vectors)[:, 1]
        if (random.random() < settings.EXPLORE_PROBABILITY):
            print("--> explore uncertian examples")
            scoreFunc = lambda x: np.abs(0.5 - x)
        else:
            print("--> exploit our best guess")
            scoreFunc = lambda x: -x
        rankedDocs = self.dither(sorted(zip(self.doc_ids, scores), key=lambda x: scoreFunc(x[1])))
        for docId, score in rankedDocs:
            if docId not in self.alreadySeen:
                candidates.append(self.docs[self.doc_index[docId]])
                print("--> returning doc with score %.6f" % score)
                if len(candidates) >= settings.BATCH_SIZE:
                    return candidates
        print("No unseen documents found")

    def handleResponse(self, docId, label):
        if docId not in self.doc_index:
            message = "WARNING: no document found with id %s" % docId
        else:
            self.model.partial_fit(self.vectors[self.doc_index[docId]], np.array([label]),
                                     classes=np.array([0, 1]), sample_weight=np.array([settings.CLASS_WEIGHTS[label]]))
            self.alreadySeen.add(docId)
            message = "Updated model with label %d for document %s" % (label, docId)
        print(message)
        return message

    def print_features(self, k):
        print("\nTop %d features" % k)
        for feature, score in self.top_features(k):
            print(feature, score)
        print("\nBottom %d features" % k)
        for feature, score in self.top_features(k):
            print(feature, score)

    def top_features(self, k):
        return [(f[0][0], f[1]) for f in sorted(zip(self.sorted_vocab, self.model.coef_[0]), key=lambda x: -x[1])[:k]]

    def bottom_features(self, k):
        return [(f[0][0], f[1]) for f in sorted(zip(self.sorted_vocab, self.model.coef_[0]), key=lambda x: x[1])[:k]]

    @staticmethod
    def dither(results):
        # TODO: implement dithering
        return results

    @staticmethod
    def loadItems(dataDir):
        docs = []
        for file in os.listdir(dataDir):
            with open(os.path.join(dataDir, file)) as f:
                for line in f:
                    rawDoc = json.loads(line)
                    docs.append(dict(
                        id=rawDoc["id"],
                        title=rawDoc["title"],
                        description=rawDoc["abstract"]
                    ))
        return docs