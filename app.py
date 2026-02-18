from flask import Flask, render_template_string, request
import spacy
from collections import Counter

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

def analyze_text(text, file_name, top_n=10):
    doc = nlp(text)

    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    avg_sentence_length = len(words) / len(sentences)

    # MDD Calculation
    mdd_list = []
    for sent in sentences:
        distances = [abs(token.i - token.head.i) for token in sent if token.is_alpha]
        if distances:
            mdd_list.append(max(distances))
    average_mdd = sum(mdd_list) / len(mdd_list)

    # Common words
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    common_words = Counter(lemmas).most_common(top_n)

    # Entities
    entity_counts = Counter([(ent.text, ent.label_) for ent in doc.ents]).most_common(top_n)
    entity_labels = [f"{ent[0]} ({ent[1]})" for ent, _ in entity_counts]

    # Pronouns
    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    pronoun_counts = Counter(pronouns).most_common(top_n)

    # Bigrams
    bigrams = zip(lemmas, lemmas[1:])
    bigram_counts = Counter(bigrams).most_common(top_n)

    # Totals
    common_words_total = sum(count for _, count in common_words)
    entities_total = sum(count for _, count in entity_counts)
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
        "entities_total": entities_total,
        "entity_labels": entity_labels,
        "pronouns": pronoun_counts,
        "pronouns_total": pronouns_total,
        "bigrams": bigram_counts,
        "bigrams_total": bigrams_total
    }

@app.route("/", methods=["GET", "POST"])
def home():
    top_n = request.args.get("top_n", default=10, type=int)
    if top_n <= 0:
        top_n = 10

    default_file = "Extract1_2016.txt"

    if request.method == "POST" and "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            text = uploaded_file.read().decode("utf-8")
            file_name = uploaded_file.filename
        else:
            with open(default_file, "r", encoding="utf-8") as f:
                text = f.read()
            file_name = default_file
    else:
        with open(default_file, "r", encoding="utf-8") as f:
            text = f.read()
        file_name = default_file

    results = analyze_text(text, file_name, top_n)

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Speech Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
</head>
<body>
<h1>Speech Analysis Results</h1>

<form method="post" enctype="multipart/form-data">
    <label>Upload a .txt speech file:</label>
    <input type="file" name="file" accept=".txt">
    <button type="submit">Analyze File</button>
</form>

<p><strong>File Analyzed:</strong> {{ results.file_name }}</p>

<form method="get">
    <label>Number of top results:</label>
    <input type="number" name="top_n" min="1" placeholder="10">
    <button type="submit">Update</button>
</form>

<p><strong>Currently showing:</strong> {{ results.top_n }}</p>

<h2>Overall Statistics</h2>

<div id="statsBlock">
    <p>Total Sentences: {{ results.sentences }}</p>
    <p>Total Words: {{ results.words }}</p>
    <p>Average Sentence Length: {{ results.avg_sentence_length }}</p>
    <p>Average MDD: {{ results.average_mdd }}</p>
</div>
                                  
<div id="totalsBlock">
    <p>Common Words: {{ results.common_words_total }}</p>
    <p>Entities: {{ results.entities_total }}</p>
    <p>Pronouns: {{ results.pronouns_total }}</p>
    <p>Common Phrases: {{ results.bigrams_total }}</p>
</div>
                                  
<hr>
                                  
<h3>View Mode:</h3>
<label>
    <input type="radio" name="viewMode" value="list" checked onclick="toggleView()"> List View
</label>
<label>
    <input type="radio" name="viewMode" value="chart" onclick="toggleView()"> Chart View
</label>
                                  
<div id="listView">
        <h2>{{ results.top_n }} Most Common Words</h2>
    <ul>
    {% for word, count in results.common_words %}
        <li>{{ word }} — {{ count }}</li>
    {% endfor %}
    </ul>

    <h2>{{ results.top_n }} Top Entities</h2>
    <ul>
    {% for ent, count in results.entities %}
        <li>{{ ent }} — {{ count }}</li>
    {% endfor %}
    </ul>
    
    <h2>{{ results.top_n }} Most Pronouns</h2>
    <ul>
    {% for pronoun, count in results.pronouns %}
        <li>{{ pronoun }} — {{ count }}</li>
    {% endfor %}
    </ul>

    <h2>{{ results.top_n }} Common Phrases</h2>
    <ul>
    {% for phrase, count in results.bigrams %}
        <li>{{ phrase[0] }} {{ phrase[1] }} — {{ count }}</li>
    {% endfor %}
    </ul>
</div>

<div id="chartView" style="display:none;">
    <!-- ALL your chart canvases go here -->
    <!-- Charts -->
    <h2>{{ results.top_n }} Most Common Words (Chart)</h2>
    <canvas id="wordsChart" width="400" height="200"></canvas>
                                  
    <hr>
                                  
    <h2>{{ results.top_n }} Top Entities (Chart)</h2>
    <canvas id="entitiesChart" width="400" height="200"></canvas>
                                  
    <hr>

    <h2>{{ results.top_n }} Most Pronouns (Chart)</h2>
    <canvas id="pronounsChart" width="400" height="200"></canvas>
                                  
    <hr>

    <h2>{{ results.top_n }} Common Phrases (Chart)</h2>
    <canvas id="bigramsChart" width="400" height="200"></canvas>
</div>

<script>
// Register the Data Labels plugin
Chart.register(ChartDataLabels);

// Convert Python data to JS
const commonWordsLabels = {{ results.common_words | map(attribute=0) | list | safe }};
const commonWordsCounts = {{ results.common_words | map(attribute=1) | list | safe }};

const pronounsLabels = {{ results.pronouns | map(attribute=0) | list | safe }};
const pronounsCounts = {{ results.pronouns | map(attribute=1) | list | safe }};

const bigramsLabels = {{ results.bigrams | map(attribute=0) | map('join', ' ') | list | safe }};
const bigramsCounts = {{ results.bigrams | map(attribute=1) | list | safe }};

const entitiesLabels = {{ results.entity_labels | safe }};
const entitiesCounts = {{ results.entities | map(attribute=1) | list | safe }};

// Chart options to show numbers on bars
const chartOptions = {
    plugins: {
        datalabels: {
            anchor: 'end',
            align: 'end',
            color: 'black',
            font: { weight: 'bold' },
            formatter: (value) => value
        }
    },
    responsive: true,
    scales: {
        y: { beginAtZero: true }
    }
};

// Words Chart
new Chart(document.getElementById('wordsChart'), {
    type: 'bar',
    data: { labels: commonWordsLabels, datasets: [{ label: 'Count', data: commonWordsCounts, backgroundColor: 'rgba(54, 162, 235, 0.6)' }] },
    options: chartOptions
});

// Pronouns Chart
new Chart(document.getElementById('pronounsChart'), {
    type: 'bar',
    data: { labels: pronounsLabels, datasets: [{ label: 'Count', data: pronounsCounts, backgroundColor: 'rgba(255, 99, 132, 0.6)' }] },
    options: chartOptions
});

// Bigrams Chart
new Chart(document.getElementById('bigramsChart'), {
    type: 'bar',
    data: { labels: bigramsLabels, datasets: [{ label: 'Count', data: bigramsCounts, backgroundColor: 'rgba(255, 206, 86, 0.6)' }] },
    options: chartOptions
});

// Entities Chart
new Chart(document.getElementById('entitiesChart'), {
    type: 'bar',
    data: { labels: entitiesLabels, datasets: [{ label: 'Count', data: entitiesCounts, backgroundColor: 'rgba(75, 192, 192, 0.6)' }] },
    options: chartOptions
});

function toggleView() {
    const selected = document.querySelector('input[name="viewMode"]:checked').value;

    if (selected === "list") {
        document.getElementById("listView").style.display = "block";
        document.getElementById("chartView").style.display = "none";
    } else {
        document.getElementById("listView").style.display = "none";
        document.getElementById("chartView").style.display = "block";
    }
}
</script>

</body>
</html>
""", results=results)

if __name__ == "__main__":
    app.run(debug=True)
