let selectedModel = "model3.pkl"; // Default model

// Toggle the model dropdown visibility
function toggleModelDropdown() {
    const dropdown = document.getElementById("model-dropdown");
    dropdown.style.display = dropdown.style.display === "none" ? "block" : "none";
}

// Update the selected model when the user chooses from the dropdown
function changeModel() {
    const modelSelect = document.getElementById("model-select");
    selectedModel = modelSelect.value;
    alert(`Model changed to: ${selectedModel}`);
}

// Function to preview the uploaded image
function previewImage(input) {
    const preview = document.getElementById("image-preview");

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = "block";
        };

        reader.readAsDataURL(input.files[0]);
    }
}

// Function to handle image upload and send request to FastAPI backend
async function predictBin() {
    const city = document.getElementById("postal-code").value;
    const fileInput = document.getElementById("image-upload");
    const file = fileInput.files[0];

    if (!city || !file) {
        alert("Please select a city and upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("city", city);
    formData.append("file", file);
    formData.append("model", selectedModel);

    try {
        const response = await fetch(`${window.location.origin}/predict-bin/`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`An error occurred: ${response.statusText}`);
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error("Error:", error);
        alert("There was an error processing your request.");
    }
}

// Function to display prediction results on the UI
function displayResults(result) {
    const resultsContainer = document.getElementById("results");
    resultsContainer.innerHTML = `
        <h3>Prediction Results</h3>
        <p><strong>Predicted Waste Type:</strong> ${result.predicted_waste_type}</p>
        <p><strong>Recommended Trash Can for ${result.city}:</strong> ${result.recommended_bin}</p>
    `;
}

// Attach event listener to the "Which Bin?" button
document.querySelector(".search-button").addEventListener("click", predictBin);
