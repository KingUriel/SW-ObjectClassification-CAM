'use strict';
window.onload = async e => {
    // Wait for TensorFlow.js to be ready
    const startLoadingTf = performance.now();
    await tf.ready();
    const endLoadingTf = performance.now();
    console.info(`TensorFlow.js loaded in ${endLoadingTf - startLoadingTf} ms`, tf.memory());

    if (navigator.userAgent.includes('Firefox')) {
        const majorVersion = parseInt(navigator.userAgent.match('Firefox\/([0-9]+)')[1]);
        if (majorVersion < 60) alert('Firefox is outdated. Please update and relaunch it.');
    } else if (navigator.userAgent.includes("Chrome")) {
        const majorVersion = parseInt(navigator.userAgent.match('Chrome\/([0-9]+)')[1]);
        if (majorVersion < 66) alert('Chrome is outdated. Please update and relaunch it.');
    } else {
        alert("This site has only been tested with Chrome and Firefox.");
    }

    if (!tf.getBackend().includes('webgl')) {
        alert("This site will run very slow without WebGL enabled. Please enable hardware acceleration and relaunch your browser.");
    }

    await populateModelSelect();
    await populateClassSelect();

    const image = await imageSearch('chauve-souris');
    drawImage(image);

    const slider = document.getElementById('opacitySlider');
    slider.onchange = slider.oninput = e => saliencyMap.style.opacity = slider.value;
    saliencyMap.style.opacity = slider.value;

    const modelSelect = document.getElementById('modelSelect');
    modelSelect.onchange = async () => {
        modelSelect.classList.add('pulse');
        modelSelect.disabled = true;

        const modelName = modelSelect.value;
        const model = await tf.loadLayersModel("../models/" + modelName + "/model.json");
        console.info("Loaded tfjs model.", model, tf.memory());

        const image = document.getElementById('imageCanvas');
        await classifyImage(image, model);
        await plotSalience(image, model);
        const classSelect = document.getElementById('classSelect');
        classSelect.onchange = async e => {
            classSelect.classList.add('pulse');
            classSelect.disabled = true;
            const image = document.getElementById('imageCanvas');
            await plotSalience(image, model);
            classSelect.classList.remove('pulse');
            classSelect.disabled = false;
        };

        const button = document.getElementById('imageButton');
        button.onclick = async e => {
            e.preventDefault();
            button.classList.add('pulse');
            button.disabled = true;
            const query = document.getElementById('searchQuery').value;
            const image = await imageSearch(query);
            drawImage(image);
            await classifyImage(image, model);
            await plotSalience(image, model);
            button.classList.remove('pulse');
            button.disabled = false;
        };

        const body = document.querySelector('body');
        body.ondragover = e => e.preventDefault();
        body.ondragend = e => e.preventDefault();
        body.ondrop = e => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            const reader = new FileReader();
            reader.onload = e => {
                const image = new Image();
                image.src = e.target.result;
                image.onload = async e => {
                    const area = document.getElementById('dropArea');
                    drawImage(image);
                    await classifyImage(image, model);
                    await plotSalience(image, model);
                };
            };
            reader.readAsDataURL(file);
        };
        modelSelect.classList.remove('pulse');
        modelSelect.disabled = false;
    };
    modelSelect.onchange();
};

function drawImage(image) {
    const imageCanvas = document.getElementById('imageCanvas');
    const saliencyMap = document.getElementById('saliencyMap');
    imageCanvas.width = image.width;
    imageCanvas.height = image.height;
    const ctx = imageCanvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    saliencyMap.width = imageCanvas.width;
    saliencyMap.height = imageCanvas.height;
    return image;
}

async function classifyImage(image, model) {
    const startClassification = performance.now();

    const x = preprocess(image, model);
    const y = model.predict(x);
    const scores = await y.data();
    
    await populateClassSelect(scores);
    const classSelect = document.getElementById('classSelect');
    document.querySelector('h1').innerHTML = classSelect[0].text;
    document.querySelector('meter').value = Math.max.apply(Math, scores);

    const endClassification = performance.now();
    console.info(`Classification took ${endClassification - startClassification} ms`, tf.memory());
}

async function plotSalience(image, model) {
    const startCAM = performance.now();
    const x = preprocess(image, model);
    const classIndex = parseInt(classSelect.value);
    const gradients = await gradCam(x, model, classIndex);
    drawSaliencyMap(gradients);
    const endCAM = performance.now();
    console.info(`Grad-CAM took ${endCAM - startCAM} ms`, tf.memory());
}

function preprocess(image, model) {
    const normalize = tensor => {
        switch (model.name) {
            case "vgg16":
            case "vgg19":
            case "resnet50":
                const normalized = tensor.sub(tf.tensor1d([
                    103.939,
                    116.779,
                    123.68]));
                const [r, g, b] = tf.split(normalized, 3, 2);
                return tf.concat([b, g, r], 2);
            default:
                return tensor.div(tf.scalar(127.5)).sub(tf.scalar(1.0));
        }
    };
    const imageTensor = tf.browser.fromPixels(image).toFloat();
    const preprocessedImage = normalize(imageTensor);
    const batch = preprocessedImage.expandDims(0);
    const inputShape = model.input.shape.slice(1, 3);
    return tf.image.resizeBilinear(batch, inputShape);
}

function imageSearch(query) {
    return new Promise((resolve, reject) => {
        $.getJSON("https://api.flickr.com/services/feeds/photos_public.gne?jsoncallback=?", {
            tags: query.split(' ').join(','),
            tagmode: "all",
            format: "json"
        }, response => {
            if (response.items.length === 0) reject('No results. Search again.');
            else {
                const image = new Image();
                const n = Math.floor(Math.random() * response.items.length);
                image.src = response.items[n]['media']['m'].replace("_m", "_b");
                image.crossOrigin = '';
                image.onload = () => resolve(image);
            }
        });
    });
}

async function drawSaliencyMap(gradients) {
    const saliencyMap = document.getElementById('saliencyMap');
    const sample = gradients.squeeze().expandDims(2);
    const shape = [saliencyMap.height, saliencyMap.width];
    const resizedGradients = tf.image.resizeBilinear(sample, shape);
    const normalizedGradients = minMaxNormalize(resizedGradients);
    const pixels = await tf.browser.toPixels(normalizedGradients);
    for (let i = 0; i < pixels.length; i += 4) {
        const color = d3.color(d3.interpolateWarm(pixels[i] / 255));
        pixels[i + 0] = color.r;
        pixels[i + 1] = color.g;
        pixels[i + 2] = color.b;
        pixels[i + 3] = pixels[i];
    }
    const imageData = new ImageData(pixels, saliencyMap.width, saliencyMap.height);
    const ctx = saliencyMap.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
}

async function populateClassSelect(scores = false) {
    const select = document.getElementById('classSelect');
    const classes = await fetch(
        '../classes.json',
        { cache: 'force-cache' }).then(x => x.json());
    select.innerHTML = '';
    if (!scores) Object.values(classes).forEach(x => {
        const option = document.createElement('option');
        option.text = x.split(',')[0];
        select.add(option);
    });
    else Array.from(scores)
        .map((x, i) => ({ score: x, index: i, name: classes[i] }))
        .sort((a, b) => b.score - a.score)
        .forEach(x => {
            const option = document.createElement('option');
            option.value = x.index;
            option.text = `${x.name.split(',')[0]} (${(100 * x.score).toFixed()}%)`;
            select.add(option);
        });
}

async function populateModelSelect() {
    const select = document.getElementById('modelSelect');
    const modelNames = await fetch('models.csv')
        .then(x => x.text())
        .then(x => x.split('\n').slice(0, -1));
    select.innerHTML = '';
    for (const modelName of modelNames) {
        const option = document.createElement('option');
        option.value = option.text = modelName;
        select.add(option);
    }
}

function minMaxNormalize(tensor) {
    const max = tensor.max();
    const min = tensor.min();
    const diff = max.sub(min);
    return tensor.sub(min).div(diff).clipByValue(0, 1);
}

async function gradCam(images, model, classIndex) {
    // Determine last 2D convolution layer.
    const layer = model.layers
        .filter(x => { try { x.output; return true } catch { return false } })
        .filter(x => x.output.shape.length == 4)
        .filter(x => x.output.shape[1] > 1 || x.output.shape[2] > 1)
        .slice(-1)[0];
    console.debug(`Last convolution layer seems to be called ${layer.name}.`, layer);
    console.assert(layer.outboundNodes.length == 1);
    const cnn = tf.model({ inputs: model.inputs, outputs: layer.output });
    // Don't assume a sequential model for the task-specific section.
    const i = model.layers.lastIndexOf(layer);
    const layers = model.layers.slice(i + 1);
    const x = tf.input({ shape: layer.output.shape.slice(1) });
    let y = x;
    for (const layer of layers) y = layer.apply(y);
    const taskSpecific = tf.model({ inputs: x, outputs: y });
    console.debug('Task specific part: ', taskSpecific.layers.map(x => x.name));
    const featureMaps = cnn.predict(images);
    const f = x => taskSpecific.predict(x).slice([0, classIndex], [images.shape[0], 1]);
    const df = tf.grad(f);
    const grads = df(featureMaps);
    const alphas = tf.mean(grads, [1, 2]);
    return alphas.mul(featureMaps).sum(-1).relu();
}