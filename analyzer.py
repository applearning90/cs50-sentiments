import nltk

class Analyzer():
    """Implements sentiment analysis."""

    def __init__(self, positives, negatives):
        """Initialize Analyzer."""

        self.positives = set()
        self.negatives = set()

        with open(positives, "r") as lines:
            for line in lines:
                if not line.startswith(";"):
                    self.positives.add(line.strip())

        with open(negatives, "r") as lines:
            for line in lines:
                if not line.startswith(";"):
                    self.negatives.add(line.strip())


    def analyze(self, text):
        """Analyze text for sentiment, returning its score."""
        score = 0

        tokenizer = nltk.tokenize.TweetTokenizer()
        tokens = tokenizer.tokenize(text)

        for x in tokens:
            if x.lower() in self.positives:
                score += 1
            if x.lower() in self.negatives:
                score -= 1

        return score
