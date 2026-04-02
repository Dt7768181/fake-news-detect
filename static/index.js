async function runDetection() {
  const text = document.getElementById("tweetInput")
  .addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runDetection();
    }
});

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
      body: JSON.stringify({ text: text }) // ✅ FIXED
    });

    if (!res.ok) {
      throw new Error("Server error");
    }

    data = await res.json();

  } catch (err) {
    alert("Server is waking up or error occurred. Please try again.");
    console.error(err);
    return; // ✅ prevent further execution
  }

  // Final result
  document.getElementById("finalResult").innerText =
    "CLASSIFIED AS " + data.final_prediction.toUpperCase();

  document.getElementById("decisionReason").innerText =
    "Decision logic: " + data.decision_reason;

  // SVM (only if exists)
  if (data.svm) {
    createBars(
      "svmResult",
      data.svm.weights,
      `Prediction: ${data.svm.pred} (${data.svm.confidence})`
    );
  } else {
    document.getElementById("svmResult").innerHTML = "Not used (short text)";
  }

  // LightGBM (only if exists)
  if (data.lightgbm) {
    createBars(
      "lgbmResult",
      data.lightgbm.weights,
      `Prediction: ${data.lightgbm.pred} (${data.lightgbm.confidence})`
    );
  } else {
    document.getElementById("lgbmResult").innerHTML = "Not used (short text)";
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