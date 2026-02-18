from flask import Flask, render_template_string, request
import spacy
from collections import Counter

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

def analyze_text(text, file_name, top_n=10):
    doc = nlp(text)

    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    avg_sentence_length = len(words) / len(sentences) if sentences else 0

    # MDD Calculation
    mdd_list = []
    for sent in sentences:
        distances = [abs(token.i - token.head.i) for token in sent if token.is_alpha]
        if distances:
            mdd_list.append(max(distances))
    average_mdd = sum(mdd_list) / len(mdd_list) if mdd_list else 0

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
    bigrams = list(zip(lemmas, lemmas[1:]))
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech Analysis Dashboard</title>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>

<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: 'Inter', sans-serif;
    background: #f4f6f9;
    color: #2c3e50;
}
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
h1 {
    text-align: center;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.chart-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}
.chart-total {
    margin-bottom: 10px;
    font-weight: 600;
    color: #2c3e50;
}                            
.grid {
    display: grid;
    gap: 20px;
}
.grid-2 {
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}
                                  
.list-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.list-card {
    flex: 1 1 48%;
    min-width: 280px;
}

.stats p { margin: 6px 0; font-weight: 500; }
form { margin-bottom: 10px; }
input, button {
    padding: 8px;
    border-radius: 6px;
}
input { border: 1px solid #ccc; }
button {
    border: none;
    background: #3b82f6;
    color: white;
    font-weight: 600;
    cursor: pointer;
}
button:hover { background: #2563eb; }
.view-toggle {
    display: flex;
    gap: 15px;
    margin: 20px 0;
}
canvas {
    width: 100% !important;
    height: 350px !important;
}
@media (max-width: 600px) {
    h1 { font-size: 22px; }
}
</style>
</head>

<body>
<div class="container">

<h1>Speech Analysis Dashboard</h1>

<div class="card">
<form method="post" enctype="multipart/form-data">
    <label>Upload a .txt speech file:</label>
    <input type="file" name="file" accept=".txt">
    <button type="submit">Analyze</button>
</form>

<form method="get">
    <label>Top Results:</label>
    <input type="number" name="top_n" min="1" value="{{ results.top_n }}">
    <button type="submit">Update</button>
</form>

<p><strong>File:</strong> {{ results.file_name }}</p>
</div>

<div class="grid grid-2">
<div class="card stats">
<h2>Overall Statistics</h2>
<p>Total Sentences: {{ results.sentences }}</p>
<p>Total Words: {{ results.words }}</p>
<p>Average Sentence Length: {{ results.avg_sentence_length }}</p>
<p>Average MDD: {{ results.average_mdd }}</p>
</div>

<div class="card stats">
<h2>Category Totals</h2>
<p>Common Words: {{ results.common_words_total }}</p>
<p>Entities: {{ results.entities_total }}</p>
<p>Pronouns: {{ results.pronouns_total }}</p>
<p>Common Phrases: {{ results.bigrams_total }}</p>
</div>
</div>

<div class="view-toggle">
<label><input type="radio" name="viewMode" value="list" checked onclick="toggleView()"> List View</label>
<label><input type="radio" name="viewMode" value="chart" onclick="toggleView()"> Chart View</label>
</div>

<div id="listView">

<div class="list-grid">

    <div class="card list-card">
        <h2>Top {{ results.top_n }} Common Words</h2>
        <ul>
        {% for word, count in results.common_words %}
            <li>{{ word }} — {{ count }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="card list-card">
        <h2>Top {{ results.top_n }} Entities</h2>
        <ul>
        {% for ent, count in results.entities %}
            <li>{{ ent }} — {{ count }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="card list-card">
        <h2>Top {{ results.top_n }} Pronouns</h2>
        <ul>
        {% for p, c in results.pronouns %}
            <li>{{ p }} — {{ c }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="card list-card">
        <h2>Top {{ results.top_n }} Common Phrases</h2>
        <ul>
        {% for phrase, c in results.bigrams %}
            <li>{{ phrase[0] }} {{ phrase[1] }} — {{ c }}</li>
        {% endfor %}
        </ul>
    </div>

</div>
</div>

<div id="chartView" style="display:none;">

    <div class="chart-card">
        <h2>Common Words</h2>
        <p class="chart-total">Total: {{ results.common_words_total }}</p>
        <canvas id="wordsChart"></canvas>
    </div>

    <div class="chart-card">
        <h2>Entities</h2>
        <p class="chart-total">Total: {{ results.entities_total }}</p>
        <canvas id="entitiesChart"></canvas>
    </div>

    <div class="chart-card">
        <h2>Pronouns</h2>
        <p class="chart-total">Total: {{ results.pronouns_total }}</p>
        <canvas id="pronounsChart"></canvas>
    </div>

    <div class="chart-card">
        <h2>Common Phrases</h2>
        <p class="chart-total">Total: {{ results.bigrams_total }}</p>
        <canvas id="bigramsChart"></canvas>
    </div>

</div>


</div>

<script>
Chart.register(ChartDataLabels);

const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        datalabels: {
            anchor: 'end',
            align: 'end',
            font: { weight: 'bold' }
        }
    },
    scales: { y: { beginAtZero: true } }
};

function createChart(id, labels, data, colour) {
    new Chart(document.getElementById(id), {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: "Count", data: data, backgroundColor: colour }] },
        options: chartOptions
    });
}

createChart("wordsChart",
    {{ results.common_words | map(attribute=0) | list | safe }},
    {{ results.common_words | map(attribute=1) | list | safe }},
    "rgba(54,162,235,0.6)"
);

createChart("entitiesChart",
    {{ results.entity_labels | safe }},
    {{ results.entities | map(attribute=1) | list | safe }},
    "rgba(75,192,192,0.6)"
);

createChart("pronounsChart",
    {{ results.pronouns | map(attribute=0) | list | safe }},
    {{ results.pronouns | map(attribute=1) | list | safe }},
    "rgba(255,99,132,0.6)"
);

createChart("bigramsChart",
    {{ results.bigrams | map(attribute=0) | map('join',' ') | list | safe }},
    {{ results.bigrams | map(attribute=1) | list | safe }},
    "rgba(255,206,86,0.6)"
);

function toggleView() {
    const selected = document.querySelector('input[name="viewMode"]:checked').value;
    document.getElementById("listView").style.display = selected === "list" ? "block" : "none";
    document.getElementById("chartView").style.display = selected === "chart" ? "block" : "none";
}
</script>

</body>
</html>
""", results=results)


if __name__ == "__main__":
    app.run(debug=True)
