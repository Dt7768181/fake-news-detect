function createBars(containerId, weights, titleText) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  const title = document.createElement("div");
  title.className = "bar-title";
  title.innerText = titleText;
  container.appendChild(title);

  if (!weights || Object.keys(weights).length === 0) {
    const empty = document.createElement("div");
    empty.className = "bar-empty";
    empty.innerText = "No important words found";
    container.appendChild(empty);
    return;
  }

  Object.entries(weights).forEach(([word, value]) => {
    const barItem = document.createElement("div");
    barItem.className = "bar-item";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.innerText = `${word} (${value.toFixed(2)})`;

    const barBg = document.createElement("div");
    barBg.className = "bar-bg";

    const bar = document.createElement("div");
    bar.className = "bar-fill";
    bar.style.width = Math.min(value * 100, 100) + "%";

    barBg.appendChild(bar);
    barItem.appendChild(label);
    barItem.appendChild(barBg);
    container.appendChild(barItem);
  });
}


async function runDetection() {
  const text = document.getElementById("tweetInput").value.trim();

  if (!text) {
    alert("Please enter a tweet");
    return;
  }

  // UI loading state
  document.getElementById("svmResult").innerHTML = "Analyzing...";
  document.getElementById("lgbmResult").innerHTML = "Analyzing...";
  document.getElementById("keywords").innerHTML = "";
  document.getElementById("finalResult").innerText = "PROCESSING...";
  document.getElementById("decisionReason").innerText = "";

  let data;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    if (!res.ok) {
      throw new Error("Server error");
    }

    data = await res.json();

  } catch (err) {
    alert("Server is waking up or error occurred. Please try again.");
    console.error(err);
    return;
  }

  // Final result
  document.getElementById("finalResult").innerText =
    "CLASSIFIED AS " + data.final_prediction.toUpperCase();

  document.getElementById("decisionReason").innerText =
    "Decision logic: " + data.decision_reason;

  // SVM
  if (data.svm) {
    createBars(
      "svmResult",
      data.svm.weights,
      `Prediction: ${data.svm.pred} (${data.svm.confidence})`
    );
  } else {
    document.getElementById("svmResult").innerHTML = "Not used";
  }

  // LightGBM
  if (data.lightgbm) {
    createBars(
      "lgbmResult",
      data.lightgbm.weights,
      `Prediction: ${data.lightgbm.pred} (${data.lightgbm.confidence})`
    );
  } else {
    document.getElementById("lgbmResult").innerHTML = "Not used";
  }

  // Keywords
  const kwBox = document.getElementById("keywords");
  kwBox.innerHTML = "";

  if (data.keywords && data.keywords.length > 0) {
    data.keywords.forEach(word => {
      const tag = document.createElement("span");
      tag.className = "keyword-tag";
      tag.innerText = word;
      kwBox.appendChild(tag);
    });
  } else {
    kwBox.innerText = "No keywords found";
  }
}


// 🔥 Enter key support (OUTSIDE function)
document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("tweetInput");

  if (textarea) {
    textarea.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        runDetection();
      }
    });
  }
});