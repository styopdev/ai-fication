// Example documents
const documents = [
  { id: 1, text: "Cozy café near the beach open late" },
  { id: 2, text: "Italian restaurant with vegan options" },
  { id: 3, text: "Modern hotel with free breakfast" },
  { id: 4, text: "Live music venue with Jazz and Blues" },
  { id: 5, text: "Quiet coworking space with fast Wi‑Fi" },
  { id: 6, text: "Family-friendly resort with pool and ocean view" },
];

let model = null;
let docEmbeddings = null; // tf.Tensor2D

// Render items
function renderItems() {
  const itemsDiv = document.getElementById("items");
  itemsDiv.innerHTML = documents
    .map(d => `<div class="item"><strong>${d.text}</strong></div>`)
    .join("");
}

// Compute and cache document embeddings
async function prepareEmbeddings() {
  const sentences = documents.map(d => d.text);
  const embeddings = await model.embed(sentences);
  docEmbeddings = embeddings;
}

// Cosine similarity using tensors
function cosineSimilarityMatrix(queryEmbedding, docEmbeddings) {
  const qNorm = tf.norm(queryEmbedding, 2, 1).expandDims(1);
  const dNorm = tf.norm(docEmbeddings, 2, 1).expandDims(1);
  const qUnit = queryEmbedding.div(qNorm);
  const dUnit = docEmbeddings.div(dNorm);
  return tf.matMul(dUnit, qUnit.transpose());
}

async function search() {
  const input = document.getElementById("searchInput").value.trim();
  const resultsDiv = document.getElementById("results");
  if (!input) {
    resultsDiv.innerHTML = "";
    return;
  }

  document.getElementById("status").textContent = "Searching…";

  const qEmbed = await model.embed([input]);
  const scores = cosineSimilarityMatrix(qEmbed, docEmbeddings);

  const scoresArr = (await scores.array()).map(row => row[0]);
  scores.dispose();
  qEmbed.dispose();

  const ranked = documents
    .map((doc, i) => ({ ...doc, score: scoresArr[i] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  resultsDiv.innerHTML = ranked
    .map(r => `
      <div class="item">
        <strong>${r.text}</strong>
        <div class="score">Score: ${r.score.toFixed(3)}</div>
      </div>
    `).join("");

  document.getElementById("status").textContent = "Done.";
}

async function init() {
  renderItems();
  document.getElementById("status").textContent = "Loading model…";
  model = await use.load();
  document.getElementById("status").textContent = "Preparing embeddings…";
  await prepareEmbeddings();
  document.getElementById("status").textContent = "Ready.";
  const btn = document.getElementById("searchBtn");
  btn.disabled = false;
  btn.addEventListener("click", search);
  document.getElementById("searchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !btn.disabled) search();
  });
}

init();
