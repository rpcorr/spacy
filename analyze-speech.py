import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# Load speech text
with open("Edited-Extract3_2025.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

sentences = list(doc.sents)
words = [t for t in doc if t.is_alpha]

# Overall statistics
print("______________________\n")
print("OVERALL STATISTICS:")
print("______________________\n")

print("Total sentences:", len(sentences))
print("Total words:", len(words))
print("Average sentence length:", f"{len(words) / len(sentences):.2f}")

# Calculate Maximum Dependency Distance (MDD) per sentence
mdd_list = []

for sent in sentences:
    distances = [abs(token.i - token.head.i) for token in sent if token.is_alpha]
    if distances:
        mdd_list.append(max(distances))

# Average MDD across all sentences
average_mdd = sum(mdd_list) / len(mdd_list)

print("\nAverage Maximum Dependency Distance (MDD):", f"{average_mdd:.2f}")

words = [
    token.lemma_.lower()
    for token in doc
    if token.is_alpha and not token.is_stop
]

common_words = Counter(words).most_common(15)

print("\n______________________")
print("\nMOST COMMON WORDS:")
print("______________________")

print("\nTotal common words:", sum(count for _, count in common_words), "\n")

for word, count in common_words:
    print(word, count)


entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_counts = Counter(entities)

print("\n______________________")
print("\nTOP ENTITIES:")
print("______________________\n")

for ent, count in entity_counts.most_common(10):
    print(ent, count)


pronouns = [
    token.text.lower()
    for token in doc
    if token.pos_ == "PRON"
]

pronoun_counts = Counter(pronouns).most_common()

print("\n______________________")
print("\nPRONOUNS:")
print("______________________")
print("\nTotal:", len(pronouns), "\n")
##print(Counter(pronouns))
for pronoun, count in pronoun_counts:
    print(pronoun, count)


bigrams = zip(words, words[1:])
bigram_counts = Counter(bigrams).most_common(10)

print("\n______________________")
print("\nCOMMON PHRASES:")
print("______________________")
print("\nTotal common phrases:", sum(count for _, count in bigram_counts), "\n")

for phrase, count in bigram_counts:
    print(" ".join(phrase), count)
