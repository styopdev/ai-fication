let model;

// Load MobileNet model once
mobilenet.load().then(m => {
  model = m;
  console.log("Model loaded!");
});

async function getEmbedding(fileInput) {
  return new Promise((resolve) => {
    const file = fileInput.files[0];
    const img = new Image();
    img.onload = async () => {
      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224,224])
        .expandDims(0);
      const embedding = model.infer(tensor, true); // shape [1,1024]
      const flat = embedding.flatten();            // shape [1024]
      resolve(flat);
    };
    img.src = URL.createObjectURL(file);
  });
}
// Show preview when file is selected
function showPreview(input, previewId) {
  const file = input.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById(previewId).src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}
// Cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.dot(b).dataSync()[0];
  const normA = a.norm().dataSync()[0];
  const normB = b.norm().dataSync()[0];
  return dot / (normA * normB);
}

// Compare 3rd image with first two
async function compareImages() {
  const emb1 = await getEmbedding(document.getElementById("img1"));
  const emb2 = await getEmbedding(document.getElementById("img2"));
  const emb3 = await getEmbedding(document.getElementById("img3"));

  const sim1 = cosineSimilarity(emb3, emb1);
  const sim2 = cosineSimilarity(emb3, emb2);

  let resultText = `Similarity with Image1: ${sim1.toFixed(3)}<br>
                    Similarity with Image2: ${sim2.toFixed(3)}<br>`;

  if (sim1 > sim2) {
    resultText += "<b>Image3 is more similar to Image1</b>";
  } else {
    resultText += "<b>Image3 is more similar to Image2</b>";
  }

  document.getElementById("result").innerHTML = resultText;
}
