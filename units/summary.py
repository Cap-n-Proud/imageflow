from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

nltk.download("punkt")
nltk.download("stopwords")


text = [
    "A person is standing near a snowboard on the stone floor next to the stone wall. Two young children are playing around a metal container on the snowy sidewalk near the house. A couple of women are preparing a metal pan on the cold sidewalk. A woman is wearing a purple jacket standing near the window in the snowy yard."
]

parser = PlaintextParser.from_string(str(text), Tokenizer("english"))
summarizer = LexRankSummarizer()

summary = summarizer(parser.document, 5)  # 5 sentences in summary

print("Original text:")
print(text)
print("\nSummary:")
for sentence in summary:
    print(sentence)

from rake_nltk import Rake

r = Rake()
r.extract_keywords_from_text(str(text))

print("Keywords:", r.get_ranked_phrases())


from gensim.models import Word2Vec


model = Word2Vec(text, min_count=1)
print("Vocabulary:", list(model.wv.vocab.keys()))

# similar_words = model.wv.most_similar("text")
# print("Similar words to 'text':", similar_words)
