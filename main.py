import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying a startup in Canada."
doc = nlp(text)

print("TOKENS")
for token in doc:
    print(f"{token.text:10} | POS={token.pos_:6} | DEP={token.dep_}")

print("\nNAMED ENTITIES")
for ent in doc.ents:
    print(f"{ent.text:10} | LABEL={ent.label_}")


print([token.lemma_ for token in doc])
