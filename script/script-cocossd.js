// DOM elements
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const uploadArea = document.getElementById('upload-area');
const loadingDiv = document.getElementById('loading');
const resultsDiv = document.getElementById('results');
const originalCanvas = document.getElementById('original-canvas');
const detectionCanvas = document.getElementById('detection-canvas');
const detectionInfo = document.getElementById('detection-info');

// Model variables
let model;
let gradModel;

// Initialize the app
async function init() {
    // Load COCO-SSD model
    loadingDiv.style.display = 'block';
    try {
        model = await cocoSsd.load();
        console.log('Model loaded successfully');
        loadingDiv.style.display = 'none';
    } catch (error) {
        console.error('Error loading model:', error);
        loadingDiv.innerHTML = '<p>Error loading model. Please refresh the page.</p>';
    }

    // Set up event listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    //uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop support
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#4CAF50';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#ccc';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect({ target: fileInput });
        }
    });
}

// Handle file selection
async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.match('image.*')) {
        alert('Please select an image file.');
        return;
    }

    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';

    const reader = new FileReader();
    reader.onload = async function (e) {
        const img = new Image();
        img.onload = async function () {
            try {
                // Display original image
                displayImage(img, originalCanvas);

                // Run object detection
                const predictions = await model.detect(img);
                console.log('Predictions:', predictions);

                // Draw detection results
                drawDetections(img, detectionCanvas, predictions);

                // Display detection info
                displayDetectionInfo(predictions);

                // Show results
                resultsDiv.style.display = 'block';
                loadingDiv.style.display = 'none';
            } catch (error) {
                console.error('Error processing image:', error);
                loadingDiv.innerHTML = '<p>Error processing image. Please try another one.</p>';
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Display the original image
function displayImage(img, canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);
}

// Draw detection boxes and labels
function drawDetections(img, canvas, predictions) {
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);

    // Font settings
    const font = '16px Arial';
    ctx.font = font;
    ctx.textBaseline = 'top';

    predictions.forEach(prediction => {
        // Draw bounding box
        const x = prediction.bbox[0];
        const y = prediction.bbox[1];
        const width = prediction.bbox[2];
        const height = prediction.bbox[3];

        ctx.strokeStyle = '#00FFFF';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw label background
        ctx.fillStyle = '#00FFFF';
        const textWidth = ctx.measureText(prediction.class).width;
        const textHeight = parseInt(font, 10);
        ctx.fillRect(x, y, textWidth + 10, textHeight + 10);

        // Draw text
        ctx.fillStyle = '#000000';
        ctx.fillText(prediction.class, x + 5, y + 5);

        // Draw score
        ctx.fillText(prediction.score.toFixed(2), x + 5, y + 25);
    });
}

// Display detection information
function displayDetectionInfo(predictions) {
    if (predictions.length === 0) {
        detectionInfo.innerHTML = '<p>No objects detected.</p>';
        return;
    }

    let html = '<p><strong>Detected Objects:</strong></p><ul>';
    predictions.forEach(pred => {
        html += `<li>${pred.class} (${(pred.score * 100).toFixed(1)}% confidence)</li>`;
    });
    html += '</ul>';

    detectionInfo.innerHTML = html;
}

function minMaxNormalize(tensor) {
    const max = tensor.max();
    const min = tensor.min();
    const diff = max.sub(min);
    return tensor.sub(min).div(diff).clipByValue(0, 1);
}


// Initialize the application when the page loads
window.addEventListener('DOMContentLoaded', init);