import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# Load speech text
with open("Extract1_2016.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

sentences = list(doc.sents)
words = [t for t in doc if t.is_alpha]

print("Total sentences:", len(sentences))
print("Total words:", len(words))
print("Average sentence length:", len(words) / len(sentences))

words = [
    token.lemma_.lower()
    for token in doc
    if token.is_alpha and not token.is_stop
]

common_words = Counter(words).most_common(15)

print("Most common words:")
for word, count in common_words:
    print(word, count)


entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_counts = Counter(entities)

print("Top entities:")
for ent, count in entity_counts.most_common(10):
    print(ent, count)


pronouns = [
    token.text.lower()
    for token in doc
    if token.pos_ == "PRON"
]

print("Total pronouns:", len(pronouns))
print(Counter(pronouns))


bigrams = zip(words, words[1:])
bigram_counts = Counter(bigrams).most_common(10)

print("Common phrases:")
for phrase, count in bigram_counts:
    print(" ".join(phrase), count)
