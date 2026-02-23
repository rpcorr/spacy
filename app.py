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
    top_n = 10
    text1 = None
    text2 = None
    file1_name = "Speech A"
    file2_name = "Speech B"
    analysis_mode = request.args.get("analysis_mode", "single")

    if request.method == "POST":
        analysis_mode = request.form.get("analysis_mode", "single")

        top_n = request.form.get("top_n", 10, type=int)
        if not top_n or top_n <= 0:
            top_n = 10

        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        if file1 and file1.filename:
            text1 = file1.read().decode("utf-8")
            file1_name = file1.filename

        if file2 and file2.filename:
            text2 = file2.read().decode("utf-8")
            file2_name = file2.filename

    single_mode = analysis_mode != "compare"

    results1 = None
    results2 = None
    comparison = None

    if text1:
        results1 = analyze_text(text1, file1_name, top_n)

        if not single_mode and text2:
            results2 = analyze_text(text2, file2_name, top_n)

            shared_vocab = results1["lemmas_set"] & results2["lemmas_set"]
            unique_1 = results1["lemmas_set"] - results2["lemmas_set"]
            unique_2 = results2["lemmas_set"] - results1["lemmas_set"]

            comparison = {
                "shared": len(shared_vocab),
                "unique_1": len(unique_1),
                "unique_2": len(unique_2)
            }

    return render_template_string(
        TEMPLATE,
        results1=results1,
        results2=results2,
        comparison=comparison,
        top_n=top_n,
        single_mode=single_mode,
        analyzed=bool(text1)
    )

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech Comparison Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>

/* ==========================
   Base Styles
   ========================== */
body {
    font-family: Arial;
    background: #f4f6f9;
    margin: 0;
    padding: 20px;
}

h1 { text-align: center; }

.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,.08);
}

.upload-card {
    max-width: 900px;
    margin: 0 auto 20px auto;
}

canvas {
    width: 100% !important;
    height: 350px !important;
}

/* ==========================
   Grid Layouts
   ========================== */
.grid {
    display: grid;
    gap: 20px;
}

.grid-2 {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

/* ==========================
   View Toggle (List / Chart)
   ========================== */
.view-toggle {
    display: flex;
    justify-content: center;
    margin: 30px 0;
    gap: 10px;
}

.view-toggle input[type="radio"] {
    display: none;
}

.view-toggle label {
    padding: 12px 28px;
    border-radius: 30px;
    background: #e0e5ec;
    cursor: pointer;
    font-weight: bold;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 4px 4px 8px rgba(0,0,0,0.1),
                -4px -4px 8px rgba(255,255,255,0.7);
}

.view-toggle input[type="radio"]:checked + label {
    background: linear-gradient(135deg, #4e73df, #1cc88a);
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    transform: scale(1.05);
}

.view-toggle label:hover {
    transform: scale(1.05);
}

/* ==========================
   Forms
   ========================== */

/* Shared form layout */
.upload-form,
.topn-form {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: flex-end;
}

/* Top N form spacing */
.topn-form {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #e0e0e0;
}

/* Form row (label + input) */
.form-row {
    display: flex;
    flex-direction: row;  /* horizontal layout for children if needed */
    align-items: center;
    gap: 10px;            /* space between label/input/radios */
    width: 100%;
    margin-bottom: 10px;
}

.form-row label {
    font-weight: bold;
    margin-bottom: 6px;
}

/* ==========================
   Buttons
   ========================== */
.primary-btn,
.secondary-btn {
    font-size: 16px;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    cursor: pointer;
    transition: 0.2s ease;
    text-decoration: none;
}

.primary-btn {
    background: #4e73df;
    color: white;
}

.primary-btn:hover {
    background: #2e59d9;
}

.secondary-btn {
    background: #6f42c1;
    color: white;
}

.secondary-btn:hover {
    background: #5936a2;
}

/* ==========================
   Inputs
   ========================== */
.topn-input {
    width: 100px;
    max-width: 120px;
    padding: 6px 10px;
}

/* ==========================
   Compact Top N Form
   ========================== */
.compact-form {
    justify-content: flex-start;
    align-items: center;
    gap: 12px;
}

.compact-form .form-row {
    flex: 0 0 auto;
    flex-direction: row;
    align-items: center;
    gap: 8px;
}

.vertical-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
    align-items: flex-start;
}

.radio-row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 20px;
    margin-bottom: 10px;

}

.radio-row label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: normal;   /* prevents bold stacking look */
    margin: 0;
}

.upload-row {
    display: flex;
    flex-direction: row;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center
}

.submit-row {
    display: flex;
    justify-content: flex-start;
}
</style>
</head>
<body>

<h1>Speech Analyze Dashboard</h1>

<div class="card upload-card">

<form method="post" enctype="multipart/form-data" class="upload-form vertical-form">

  <!-- ROW 1: Mode Selection -->
    <div class="form-row radio-row">
        <label>
            <input type="radio" name="analysis_mode" value="single" {% if single_mode %}checked{% endif %}>
            Analyze One Speech
        </label>

        <label>
            <input type="radio" name="analysis_mode" value="compare" {% if not single_mode %}checked{% endif %}>
            Compare Two Speeches
        </label>
    </div>

    <!-- ROW 2: File Uploads -->
<div class="form-row upload-row">
    <div>
        <label>Upload Speech A</label>
        <input type="file" name="file1" accept=".txt" required>
    </div>

    <div id="file2Row" style="display: {% if single_mode %}none{% else %}block{% endif %};">
        <label>Upload Speech B</label>
        <input type="file" name="file2" accept=".txt" {% if not single_mode %}required{% endif %}>
    </div>

</div>

<script>
// Toggle the second upload field based on selected mode
function toggleUploadMode() {
    const mode = document.querySelector('input[name="analysis_mode"]:checked').value;
    const file2Row = document.getElementById("file2Row");
    const file2Input = file2Row.querySelector('input[name="file2"]');
    const resultsSection = document.getElementById("resultsSection");
    const clearButton = document.getElementById("clearButton");

    if (mode === "compare") {
        file2Row.style.display = "block";
        file2Input.required = true;
    } else {
        file2Row.style.display = "none";
        file2Input.required = false;
        file2Input.value = "";
    }

    // Hide previous results whenever mode changes
    if (resultsSection) {
        resultsSection.style.display = "none";
        clearButton.style.display = "none";
    }
}

// Show results when form is submitted
function showResultsOnSubmit() {
    const resultsSection = document.getElementById("resultsSection");
    const clearButton = document.getElementById("clearButton");
    if (resultsSection) {
        resultsSection.style.display = "block";
        clearButton.style.display = "block";
    }
}

// Attach toggle to radio buttons
document.querySelectorAll('input[name="analysis_mode"]').forEach(radio => {
    radio.addEventListener('change', toggleUploadMode);
});

// Attach submit event to the form
document.querySelector('form.upload-form').addEventListener('submit', showResultsOnSubmit);

// Run on page load
window.addEventListener("load", () => {
    toggleUploadMode();

    // Make sure results stay visible if page was analyzed
    const resultsSection = document.getElementById("resultsSection");
    const clearButton = document.getElementById("clearButton");
    {% if analyzed %}
    if (resultsSection) {
        resultsSection.style.display = "block";
        clearButton.style.display = "block";
    }
    {% endif %}
});
</script>

  <!-- ROW 3: Top N -->
    <div class="form-row">
        <label>Top N</label>
        <input type="number" name="top_n" value="{{ top_n }}" min="1" class="topn-input">
    </div>

    <!-- ROW 4: Submit Button -->
    <div class="form-row submit-row" style="gap:10px;">
        <button type="submit" class="primary-btn">Analyze</button>
    </div>
</form>

{% if analyzed %}
<form method="get" class="upload-form" style="display:inline;">
    <input type="hidden" name="analysis_mode" value="{{ 'compare' if not single_mode else 'single' }}">
    <button type="submit" id="clearButton" class="secondary-btn">Clear</button>
</form>
{% endif %}

</div>

{% if analyzed %}
    <div id="resultsSection">
    <div class="view-toggle">
        <input type="radio" id="listMode" name="viewMode" value="list" checked onclick="toggleView()">
        <label for="listMode">📄 List View</label>

        <input type="radio" id="chartMode" name="viewMode" value="chart" onclick="toggleView()">
        <label for="chartMode">📊 Chart View</label>
    </div>

    <div id="listView">

    <div class="grid {% if not single_mode %}grid-2{% endif %}">

    <div class="card">
    <h2>{{ results1.file_name }}</h2>
    <p>Sentences: {{ results1.sentences }}</p>
    <p>Words: {{ results1.words }}</p>
    <p>Avg Sentence Length: {{ results1.avg_sentence_length }}</p>
    <p>Avg MDD: {{ results1.average_mdd }}</p>
    </div>

    {% if not single_mode %}
        <div class="card">
            <h2>{{ results2.file_name }}</h2>
            <p>Sentences: {{ results2.sentences }}</p>
            <p>Words: {{ results2.words }}</p>
            <p>Avg Sentence Length: {{ results2.avg_sentence_length }}</p>
            <p>Avg MDD: {{ results2.average_mdd }}</p>
        </div>
    {% endif %}

    </div>

    {% if not single_mode %}
        <div class="card">
            <h2>Vocabulary Comparison</h2>
            <p>Shared Vocabulary: {{ comparison.shared }}</p>
            <p>Unique to {{ results1.file_name }}: {{ comparison.unique_1 }}</p>
            <p>Unique to {{ results2.file_name }}: {{ comparison.unique_2 }}</p>
        </div>
    {% endif %}



    <!-- TOP ANALYSIS SECTIONS SIDE BY SIDE -->
    <div class="grid {% if not single_mode %}grid-2{% endif %}">

    <!-- COMMON WORDS CARD -->
    <div class="card">
    <h3>Top {{ top_n }} Common Words</h3>
    <div style="display:flex; gap:40px;">
    <div>
    <strong>{{ results1.file_name }}</strong>
    <ul>
    {% for word, count in results1.common_words %}
    <li>{{ word }} — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>

    {% if not single_mode %}
        <div>
        <strong>{{ results2.file_name }}</strong>
        <ul>
        {% for word, count in results2.common_words %}
        <li>{{ word }} — {{ count }}</li>
        {% endfor %}
        </ul>
        </div>
    {% endif %}
    </div>
    </div>


    <!-- ENTITIES CARD -->
    <div class="card">
    <h3>Top {{ top_n }} Entities</h3>
    <div style="display:flex; gap:40px;">
    <div>
    <strong>{{ results1.file_name }}</strong>
    <ul>
    {% for ent, count in results1.entities %}
    <li>{{ ent[0] }} ({{ ent[1] }}) — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>

    {% if not single_mode %}
    <div>
    <strong>{{ results2.file_name }}</strong>
    <ul>
    {% for ent, count in results2.entities %}
    <li>{{ ent[0] }} ({{ ent[1] }}) — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>
    {% endif %}
    </div>
    </div>


    <!-- PRONOUNS CARD -->
    <div class="card">
    <h3>Top {{ top_n }} Pronouns</h3>
    <div style="display:flex; gap:40px;">
    <div>
    <strong>{{ results1.file_name }}</strong>
    <ul>
    {% for word, count in results1.pronouns %}
    <li>{{ word }} — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>

    {% if not single_mode %}
    <div>
    <strong>{{ results2.file_name }}</strong>
    <ul>
    {% for word, count in results2.pronouns %}
    <li>{{ word }} — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>
    {% endif %}

    </div>
    </div>


    <!-- COMMON PHRASES CARD -->
    <div class="card">
    <h3>Top {{ top_n }} Common Phrases (Bigrams)</h3>
    <div style="display:flex; gap:40px;">
    <div>
    <strong>{{ results1.file_name }}</strong>
    <ul>
    {% for phrase, count in results1.bigrams %}
    <li>{{ phrase[0] }} {{ phrase[1] }} — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>

    {% if not single_mode %}
    <div>
    <strong>{{ results2.file_name }}</strong>
    <ul>
    {% for phrase, count in results2.bigrams %}
    <li>{{ phrase[0] }} {{ phrase[1] }} — {{ count }}</li>
    {% endfor %}
    </ul>
    </div>
    {% endif %}
    </div>
    </div>

    </div>

    </div>


    <div id="chartView" style="display:none;">

    <div class="grid {% if not single_mode %}grid-2{% endif %}">

    <div class="card">
    <h2>{{ results1.file_name }}</h2>
    <p>Sentences: {{ results1.sentences }}</p>
    <p>Words: {{ results1.words }}</p>
    <p>Avg Sentence Length: {{ results1.avg_sentence_length }}</p>
    <p>Avg MDD: {{ results1.average_mdd }}</p>
    </div>

    {% if not single_mode %}
    <div class="card">
    <h2>{{ results2.file_name }}</h2>
    <p>Sentences: {{ results2.sentences }}</p>
    <p>Words: {{ results2.words }}</p>
    <p>Avg Sentence Length: {{ results2.avg_sentence_length }}</p>
    <p>Avg MDD: {{ results2.average_mdd }}</p>
    </div>
    {% endif %}

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
    <h2>Common Phrases (Bigrams)</h2>
    <canvas id="bigramsChart"></canvas>
    </div>

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
    {label:"{{ results1.file_name }}", data:data1, backgroundColor:"rgba(54,162,235,0.6)"}
    {% if not single_mode %}
    ,
    {label:"{{ results2.file_name }}", data:data2, backgroundColor:"rgba(255,99,132,0.6)"}
    {% endif %}
    ]
    },
    options:chartOptions
    });
    }

    createGroupedChart("statsChart",
    ["Sentences","Words"],
    [{{ results1.sentences }},{{ results1.words }}]
    {% if not single_mode and results2 %}
    ,
    [{{ results2.sentences }},{{ results2.words }}]
    {% else %}
    ,
    []
    {% endif %}
    );

    createGroupedChart("wordsChart",
    {{ results1.common_words | map(attribute=0) | list | safe }},
    {{ results1.common_words | map(attribute=1) | list | safe }}
    {% if not single_mode and results2 %}
    ,
    {{ results2.common_words | map(attribute=1) | list | safe }}
    {% else %}
    ,
    []
    {% endif %}
    );

    createGroupedChart("entitiesChart",
    {{ results1.entity_labels | safe }},
    {{ results1.entities | map(attribute=1) | list | safe }}
    {% if not single_mode and results2 %}
    ,
    {{ results2.entities | map(attribute=1) | list | safe }}
    {% else %}
    ,
    []
    {% endif %}
    );

    createGroupedChart("pronounsChart",
    {{ results1.pronouns | map(attribute=0) | list | safe }},
    {{ results1.pronouns | map(attribute=1) | list | safe }}
    {% if not single_mode and results2 %}
    ,
    {{ results2.pronouns | map(attribute=1) | list | safe }}
    {% else %}
    ,
    []
    {% endif %}
    );


    const bigramLabels={{ results1.bigrams | map(attribute=0) | map('join',' ') | list | safe }};
    const data1={{ results1.bigrams | map(attribute=1) | list | safe }};
    {% if not single_mode and results2 %}
        const data2={{ results2.bigrams | map(attribute=1) | list | safe }};
    {% else %}
        const data2=[];
    {% endif %}

    const maxVal=Math.max(...data1,...data2);

    new Chart(document.getElementById("bigramsChart"),{
    type:'bar',
    data:{labels:bigramLabels,
    datasets:[
    {label:"{{ results1.file_name }}", data:data1, backgroundColor:"rgba(54,162,235,0.6)"}
    {% if not single_mode %}
    ,
    {label:"{{ results2.file_name }}", data:data2, backgroundColor:"rgba(255,99,132,0.6)"}
    {% endif %}
    ]
    },
    options:{...chartOptions,
    scales:{y:{beginAtZero:true,suggestedMax:maxVal+0.5}}}
    });

    function toggleView(){
    const selected=document.querySelector('input[name="viewMode"]:checked').value;
    document.getElementById("listView").style.display=selected==="list"?"block":"none";
    document.getElementById("chartView").style.display=selected==="chart"?"block":"none";
    }
    </script>

{% endif %}


</body>
</html>
"""

if __name__ == "__main__":
    app.run()
