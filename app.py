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

    # MDD
    mdd_list = []
    for sent in sentences:
        distances = [abs(token.i - token.head.i) for token in sent if token.is_alpha]
        if distances:
            mdd_list.append(max(distances))
    average_mdd = sum(mdd_list) / len(mdd_list) if mdd_list else 0

    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    common_words = Counter(lemmas).most_common(top_n)
    entity_counts = Counter([(ent.text, ent.label_) for ent in doc.ents]).most_common(top_n)
    entity_labels = [f"{ent[0]} ({ent[1]})" for ent, _ in entity_counts]

    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    pronoun_counts = Counter(pronouns).most_common(top_n)

    bigrams = list(zip(lemmas, lemmas[1:]))
    bigram_counts = Counter(bigrams).most_common(top_n)

    return {
        "file_name": file_name,
        "sentences": len(sentences),
        "words": len(words),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "average_mdd": round(average_mdd, 2),
        "common_words": common_words,
        "entities": entity_counts,
        "entity_labels": entity_labels,
        "pronouns": pronoun_counts,
        "bigrams": bigram_counts,
        "lemmas_set": set(lemmas)
    }


@app.route("/", methods=["GET", "POST"])
def home():
    top_n = request.args.get("top_n", default=10, type=int)
    if top_n <= 0:
        top_n = 10

    default_file = "Extract1_2016.txt"

    text1 = None
    text2 = None
    file1_name = "Speech A"
    file2_name = "Speech B"

    if request.method == "POST":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        if file1 and file1.filename:
            text1 = file1.read().decode("utf-8")
            file1_name = file1.filename

        if file2 and file2.filename:
            text2 = file2.read().decode("utf-8")
            file2_name = file2.filename

    if not text1:
        with open(default_file, "r", encoding="utf-8") as f:
            text1 = f.read()
        file1_name = default_file

    if not text2:
        text2 = text1
        file2_name = file1_name

    results1 = analyze_text(text1, file1_name, top_n)
    results2 = analyze_text(text2, file2_name, top_n)

    shared_vocab = results1["lemmas_set"] & results2["lemmas_set"]
    unique_1 = results1["lemmas_set"] - results2["lemmas_set"]
    unique_2 = results2["lemmas_set"] - results1["lemmas_set"]

    comparison = {
        "shared": len(shared_vocab),
        "unique_1": len(unique_1),
        "unique_2": len(unique_2)
    }

    return render_template_string(TEMPLATE,
                                  results1=results1,
                                  results2=results2,
                                  comparison=comparison,
                                  top_n=top_n)


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech Comparison Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
body { font-family: Arial; background:#f4f6f9; margin:0; padding:20px; }
.card { background:white; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow:0 4px 10px rgba(0,0,0,.08); }
.grid { display:grid; gap:20px; }
.grid-2 { grid-template-columns: repeat(auto-fit,minmax(300px,1fr)); }
canvas { width:100% !important; height:350px !important; }
</style>
</head>
<body>

<h1>Speech Comparison Dashboard</h1>

<div class="card">
<form method="post" enctype="multipart/form-data">
Upload Speech A: <input type="file" name="file1" accept=".txt">
Upload Speech B: <input type="file" name="file2" accept=".txt">
<button type="submit">Compare</button>
</form>

<form method="get">
Top N: <input type="number" name="top_n" value="{{ top_n }}" min="1">
<button type="submit">Update</button>
</form>
</div>

<div style="margin:20px 0;">
<label><input type="radio" name="viewMode" value="list" checked onclick="toggleView()"> List View</label>
<label><input type="radio" name="viewMode" value="chart" onclick="toggleView()"> Chart View</label>
</div>

<div id="listView">

<div class="grid grid-2">

<div class="card">
<h2>{{ results1.file_name }}</h2>
<p>Sentences: {{ results1.sentences }}</p>
<p>Words: {{ results1.words }}</p>
<p>Avg Sentence Length: {{ results1.avg_sentence_length }}</p>
<p>Avg MDD: {{ results1.average_mdd }}</p>
</div>

<div class="card">
<h2>{{ results2.file_name }}</h2>
<p>Sentences: {{ results2.sentences }}</p>
<p>Words: {{ results2.words }}</p>
<p>Avg Sentence Length: {{ results2.avg_sentence_length }}</p>
<p>Avg MDD: {{ results2.average_mdd }}</p>
</div>

</div>

<div class="card">
<h2>Vocabulary Comparison</h2>
<p>Shared Vocabulary: {{ comparison.shared }}</p>
<p>Unique to {{ results1.file_name }}: {{ comparison.unique_1 }}</p>
<p>Unique to {{ results2.file_name }}: {{ comparison.unique_2 }}</p>
</div>

</div>

<div id="chartView" style="display:none;">

<div class="grid grid-2">

<div class="card">
<h2>{{ results1.file_name }}</h2>
<p>Sentences: {{ results1.sentences }}</p>
<p>Words: {{ results1.words }}</p>
<p>Avg Sentence Length: {{ results1.avg_sentence_length }}</p>
<p>Avg MDD: {{ results1.average_mdd }}</p>
</div>

<div class="card">
<h2>{{ results2.file_name }}</h2>
<p>Sentences: {{ results2.sentences }}</p>
<p>Words: {{ results2.words }}</p>
<p>Avg Sentence Length: {{ results2.avg_sentence_length }}</p>
<p>Avg MDD: {{ results2.average_mdd }}</p>
</div>

</div>

<div class="card">
<h2>Sentence & Word Comparison</h2>
<canvas id="statsChart"></canvas>
</div>

<div class="card">
<h2>Common Words</h2>
<canvas id="wordsChart"></canvas>
</div>

<div class="card">
<h2>Entities</h2>
<canvas id="entitiesChart"></canvas>
</div>

<div class="card">
<h2>Pronouns</h2>
<canvas id="pronounsChart"></canvas>
</div>

<div class="card">
<h2>Bigrams</h2>
<canvas id="bigramsChart"></canvas>
</div>

</div>

<script>
Chart.register(ChartDataLabels);

const chartOptions = {
responsive:true,
maintainAspectRatio:false,
plugins:{ datalabels:{ anchor:'end', align:'end', font:{weight:'bold'} } },
scales:{ y:{ beginAtZero:true } }
};

function createGroupedChart(id, labels, data1, data2){
new Chart(document.getElementById(id),{
type:'bar',
data:{
labels:labels,
datasets:[
{label:"{{ results1.file_name }}", data:data1, backgroundColor:"rgba(54,162,235,0.6)"},
{label:"{{ results2.file_name }}", data:data2, backgroundColor:"rgba(255,99,132,0.6)"}
]},
options:chartOptions
});
}

createGroupedChart("statsChart",
["Sentences","Words"],
[{{ results1.sentences }},{{ results1.words }}],
[{{ results2.sentences }},{{ results2.words }}]
);

createGroupedChart("wordsChart",
{{ results1.common_words | map(attribute=0) | list | safe }},
{{ results1.common_words | map(attribute=1) | list | safe }},
{{ results2.common_words | map(attribute=1) | list | safe }}
);

createGroupedChart("entitiesChart",
{{ results1.entity_labels | safe }},
{{ results1.entities | map(attribute=1) | list | safe }},
{{ results2.entities | map(attribute=1) | list | safe }}
);

createGroupedChart("pronounsChart",
{{ results1.pronouns | map(attribute=0) | list | safe }},
{{ results1.pronouns | map(attribute=1) | list | safe }},
{{ results2.pronouns | map(attribute=1) | list | safe }}
);

const bigramLabels={{ results1.bigrams | map(attribute=0) | map('join',' ') | list | safe }};
const data1={{ results1.bigrams | map(attribute=1) | list | safe }};
const data2={{ results2.bigrams | map(attribute=1) | list | safe }};
const maxVal=Math.max(...data1,...data2);

new Chart(document.getElementById("bigramsChart"),{
type:'bar',
data:{labels:bigramLabels,
datasets:[
{label:"{{ results1.file_name }}",data:data1,backgroundColor:"rgba(54,162,235,0.6)"},
{label:"{{ results2.file_name }}",data:data2,backgroundColor:"rgba(255,99,132,0.6)"}
]},
options:{...chartOptions,
scales:{y:{beginAtZero:true,suggestedMax:maxVal+0.5}}}
});

function toggleView(){
const selected=document.querySelector('input[name="viewMode"]:checked').value;
document.getElementById("listView").style.display=selected==="list"?"block":"none";
document.getElementById("chartView").style.display=selected==="chart"?"block":"none";
}
</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
