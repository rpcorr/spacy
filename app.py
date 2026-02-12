from flask import Flask, render_template_string
import spacy
from collections import Counter

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

top_n = 10

def analyze_text():
    file_name = "Edited-Extract3_2025.txt"

    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)

    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)

    # MDD Calculation
    mdd_list = []
    for sent in sentences:
        distances = [abs(token.i - token.head.i) for token in sent if token.is_alpha]
        if distances:
            mdd_list.append(max(distances))

    average_mdd = sum(mdd_list) / len(mdd_list)

    # Common words
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    common_words = Counter(lemmas).most_common(top_n)

    # Entities
    entity_counts = Counter([(ent.text, ent.label_) for ent in doc.ents]).most_common(top_n)

    # Pronouns
    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    pronoun_counts = Counter(pronouns).most_common(top_n)

    # Bigrams
    bigrams = zip(lemmas, lemmas[1:])
    bigram_counts = Counter(bigrams).most_common(top_n)

    # Sum of counts
    common_words_total = sum(count for _, count in common_words)
    entity_counts_total = sum(count for _, count in entity_counts)
    pronouns_total = sum(count for _, count in pronoun_counts)
    bigrams_total = sum(count for _, count in bigram_counts)

    return {
        "file_name": file_name,
        "top_n": top_n,
        "sentences": len(sentences),
        "words": len(words),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "average_mdd": round(average_mdd, 2),
        "common_words": common_words,
        "common_words_total": common_words_total,
        "entities": entity_counts,
        "entities_total": entity_counts_total,
        "pronouns": pronoun_counts,
        "pronouns_total": pronouns_total,
        "bigrams": bigram_counts,
        "bigrams_total": bigrams_total,
        "bigrams_total": bigrams_total
    }
    

@app.route("/")
def home():
    results = analyze_text()

    return render_template_string("""
    <h1>Speech Analysis Results</h1>
                                  
    <p><strong>File Analyzed:</strong> {{ results.file_name }}</p>

    <h2>Overall Statistics</h2>
    <p>Total Sentences: {{ results.sentences }}</p>
    <p>Total Words: {{ results.words }}</p>
    <p>Average Sentence Length: {{ results.avg_sentence_length }}</p>
    <p>Average MDD: {{ results.average_mdd }}</p>

    <h2>{{ results.top_n }} Most Common Words</h2>
    <ul>
    {% for word, count in results.common_words %}
        <li>{{ word }} — {{ count }}</li>
    {% endfor %}
    </ul>
                                  
    <p><strong>Total Common Words:</strong> {{ results.common_words_total }}</p>

    <h2>{{ results.top_n }} Top Entities</h2>
    <ul>
    {% for ent, count in results.entities %}
        <li>{{ ent }} — {{ count }}</li>
    {% endfor %}
    </ul>
                                  
    <p><strong>Total Entities:</strong> {{ results.entities_total }}</p>

    <h2>{{ results.top_n }} Most Pronouns</h2>
    <ul>
    {% for pronoun, count in results.pronouns %}
        <li>{{ pronoun }} — {{ count }}</li>
    {% endfor %}
    </ul>
                                  
    <p><strong>Total Pronouns:</strong> {{ results.pronouns_total }}</p>

    <h2>{{ results.top_n }} Common Phrases</h2>
    <ul>
    {% for phrase, count in results.bigrams %}
        <li>{{ phrase[0] }} {{ phrase[1] }} — {{ count }}</li>
    {% endfor %}
    </ul>
                                  
    <p><strong>Total Common Phrases:</strong> {{ results.bigrams_total }}</p>
    """, results=results)

if __name__ == "__main__":
    app.run(debug=True)
