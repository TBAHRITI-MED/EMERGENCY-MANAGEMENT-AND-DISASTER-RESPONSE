// script.js
function uploadImage() {
  var fileInput = document.getElementById("image-input");
  var file = fileInput.files[0];

  var formData = new FormData();
  formData.append("image", file);

  fetch("localhost:8000/predict/", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      var predictionElement = document.getElementById("prediction");
      predictionElement.textContent = `Class: ${data.class}, Confidence: ${data.confidence}`;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
