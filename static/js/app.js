document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const cameraInput = document.getElementById("camera-input");
    const btnBrowse = document.getElementById("btn-browse");
    const btnCamera = document.getElementById("btn-camera");
    const btnRun = document.getElementById("btn-run");
    
    const confSlider = document.getElementById("conf-slider");
    const confVal = document.getElementById("conf-val");
    
    const inputPreviewBox = document.getElementById("input-preview-box");
    const inputPreview = document.getElementById("input-preview");
    
    const emptyState = document.getElementById("empty-state");
    const spinner = document.getElementById("spinner");
    const emptyIcon = document.getElementById("empty-icon");
    const statusText = document.getElementById("status-text");
    
    const resultsContent = document.getElementById("results-content");
    const annotatedImage = document.getElementById("annotated-image");
    const cropImage = document.getElementById("crop-image");
    const plateTextDisplay = document.getElementById("plate-text-display");
    const valTime = document.getElementById("val-time");
    const valConf = document.getElementById("val-conf");
    
    const selectorRow = document.getElementById("selector-row");
    const plateSelect = document.getElementById("plate-select");

    // State variables
    let selectedFile = null;
    let detectionResults = [];

    // --- Slider Event ---
    confSlider.addEventListener("input", (e) => {
        const val = (e.target.value / 100).toFixed(2);
        confVal.textContent = val;
    });

    // --- File Input Actions ---
    btnBrowse.addEventListener("click", () => fileInput.click());
    
    // Trigger mobile native camera capture
    btnCamera.addEventListener("click", () => cameraInput.click());

    fileInput.addEventListener("change", (e) => handleFileSelection(e.target.files[0]));
    cameraInput.addEventListener("change", (e) => handleFileSelection(e.target.files[0]));

    // --- Drag and Drop Events ---
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelection(file) {
        if (!file) return;

        // Check if file is image
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            return;
        }

        selectedFile = file;
        
        // Show local preview
        const reader = new FileReader();
        reader.onload = (e) => {
            inputPreview.src = e.target.result;
            inputPreviewBox.classList.remove("hidden");
            btnRun.disabled = false;
        };
        reader.readAsDataURL(file);

        // Reset output state
        resetOutputState();
    }

    function resetOutputState() {
        resultsContent.classList.add("hidden");
        emptyState.classList.remove("hidden");
        spinner.classList.add("hidden");
        emptyIcon.classList.remove("hidden");
        emptyIcon.textContent = "🤖";
        statusText.textContent = "Image selected. Press 'Run Recognition' to start.";
    }

    // --- Run Recognition API ---
    btnRun.addEventListener("click", async () => {
        if (!selectedFile) return;

        // Show loading state
        btnRun.disabled = true;
        btnBrowse.disabled = true;
        btnCamera.disabled = true;
        
        emptyIcon.classList.add("hidden");
        spinner.classList.remove("hidden");
        statusText.textContent = "Processing image (YOLOv8 & PaddleOCR)...";

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("conf", (confSlider.value / 100).toFixed(2));

        try {
            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Server error occurred.");
            }

            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
            } else {
                showErrorState(data.error);
            }

        } catch (error) {
            console.error(error);
            showErrorState(error.message);
        } finally {
            btnRun.disabled = false;
            btnBrowse.disabled = false;
            btnCamera.disabled = false;
        }
    });

    function showErrorState(message) {
        resultsContent.classList.add("hidden");
        emptyState.classList.remove("hidden");
        spinner.classList.add("hidden");
        emptyIcon.classList.remove("hidden");
        emptyIcon.textContent = "⚠️";
        statusText.textContent = "Error: " + message;
    }

    function displayResults(data) {
        emptyState.classList.add("hidden");
        resultsContent.classList.remove("hidden");

        // 1. Set annotated image
        annotatedImage.src = data.annotated_image;
        
        // Save results globally
        detectionResults = data.plates;
        valTime.textContent = data.elapsed_time + " s";

        // 2. Clear selector dropdown
        plateSelect.innerHTML = "";

        if (detectionResults.length === 0) {
            selectorRow.classList.add("hidden");
            cropImage.src = "";
            cropImage.alt = "No Crop";
            plateTextDisplay.textContent = "NONE";
            valConf.textContent = "0%";
            document.getElementById("vn-plate-display").style.backgroundColor = "#ffebee"; // red alert tint
        } else {
            document.getElementById("vn-plate-display").style.backgroundColor = "#f8fafc";
            
            // Populate dropdown if multiple plates detected
            if (detectionResults.length > 1) {
                selectorRow.classList.remove("hidden");
                detectionResults.forEach((res, index) => {
                    const textPreview = res.text || "Empty text";
                    const opt = document.createElement("option");
                    opt.value = index;
                    opt.textContent = `Plate ${index + 1}: ${textPreview}`;
                    plateSelect.appendChild(opt);
                });
            } else {
                selectorRow.classList.add("hidden");
            }

            // Show first result by default
            renderSinglePlateResult(0);
        }
    }

    // --- Dropdown select trigger ---
    plateSelect.addEventListener("change", (e) => {
        renderSinglePlateResult(parseInt(e.target.value));
    });

    function renderSinglePlateResult(index) {
        if (index < 0 || index >= detectionResults.length) return;
        
        const plate = detectionResults[index];
        cropImage.src = plate.crop_image;
        
        if (plate.text) {
            plateTextDisplay.textContent = plate.text;
        } else {
            plateTextDisplay.textContent = "EMPTY";
        }
        
        valConf.textContent = plate.confidence + "%";
    }
});
