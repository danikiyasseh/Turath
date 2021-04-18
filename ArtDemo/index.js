const path_to_model = './ArtMobileNetV2/model.json' 

// import {IMAGENET_CLASSES} from './imagenet_classes';

// let net;
// const TOPK_PREDICTIONS = 5

const topk = 3;
const IMAGE_HEIGHT = 224;
const IMAGE_WIDTH = 224;
let model;

const app = async () => {
  status('Loading model...');
  //console.log('Loading model..');
  
  // Load the model.
  // net = await mobilenet.load();
  model = await tf.loadLayersModel(path_to_model);
  status('Successfully loaded model!');
  // console.log('Successfully loaded model');
  
  // Forward pass with zeros to warm-up model (faster processing later).
  model.predict(tf.zeros([1,IMAGE_HEIGHT,IMAGE_WIDTH,3])).dispose();
  
  // Get image.
  const imgEl = document.getElementById('img');
  // If image available, predict. Otherwise, wait for loaded image.
  if (imgEl.complete && imgEl.naturalHeight !== 0) {
    console.log('Found image...');
    predict(imgEl, topk);
    imgEl.style.display = '';
  } else {
    console.log('waiting for image...');
    imgEl.onload = () => {
      predict(imgEl, topk);
      imgEl.style.display = '';
    }
  }  
  // Show image that was just loaded
  document.getElementById('file-container').style.display = '';
  
//   // Make a prediction through the model on our image.
//   // const result = await net.classify(imgEl);
//   const logits = await model.predict(imgSample);
//   console.log(logits);
}

async function predict(imgEl, topk) {  
  const preds = await classify(imgEl, topk)
  
  status('Done!');
  console.log(preds);
  
  showResults(imgEl, preds)
  console.log('Showed Results') 
}

// Function to make prediction and obtain topk results.
async function classify(imgEl, topk) {
    status('Predicting...');
  
    const logits = tf.tidy(() => {
          // Load image into TFJS world.
          const img = tf.browser.fromPixels(imgEl).toFloat();
          // Normalize the image from [0-255] to [0-1].
          const offset = 255;
          const normalized = img.div(offset);
          // Reshape image for network.
          const imgSample = normalized.reshape([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]);

          return model.predict(imgSample);
          //logits.dispose();
          //return classes;
    });
  
    const classes = await getTopKClasses(logits, topk);
    return classes;
}

async function getTopKClasses(logits, topK) {
  const softmax = tf.softmax(logits);
  const values = await softmax.data();
  softmax.dispose();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: artClasses[topkIndices[i]],
      probability: topkValues[i]
    });
  }
  return topClassesAndProbs;
}

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    console.log('Hello')
    // Only process image files (skip non image files)
    // if (!f.type.match('image.*')) {
    //   console.log('skipped');
    //   continue;
    // }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_WIDTH;
      img.height = IMAGE_HEIGHT;
      //console.log(img.height);
      img.onload = () => predict(img, topk);
      //console.log(img.onload);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
app();
