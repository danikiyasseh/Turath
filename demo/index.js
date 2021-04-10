const path_to_model = './MobileNetV2/model.json' 

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
      className: UNESCO_CLASSES[topkIndices[i]],
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

const UNESCO_CLASSES = {
 0: 'Abu-mena',
 1: 'Aflaj',
 2: 'Ahwar-of-southern-iraq',
 3: 'Al-ahsa-oasis',
 4: 'Al-ain',
 5: 'Al-balad,-jeddah',
 6: 'Al-maghtas',
 7: 'Al-zubarah',
 8: 'Amphitheatre-of-el-jem',
 9: 'Ancient-city-of-bosra',
 10: 'Ancient-city-of-damascus',
 11: 'Ksour-of-Ouadane',
 12: 'Anjar,-lebanon',
 13: 'Archaeological-site-of-carthage',
 14: 'Al-Khutm-and-Al-Ayn',
 15: 'Assur',
 16: 'Baalbek',
 17: 'Babylon',
 18: 'Bahla-fort',
 19: 'Bahrain-pearling-trail',
 20: 'Battir',
 21: 'Beni-hammad-fort',
 22: 'Byblos',
 23: 'Casbah-of-algiers',
 24: 'Cedars-of-god',
 25: 'Church-of-the-nativity',
 26: 'Citadel-of-arbil',
 27: 'Citadel-of-salah-ed-din',
 28: 'Cyrene,-libya',
 29: 'Dead-cities',
 30: 'Dilmun-burial-mounds',
 31: 'Diriyah',
 32: 'Djémila',
 33: 'Dougga',
 34: 'El-jadida',
 35: 'Essaouira',
 36: 'Fes-el-bali',
 37: 'Frankincense-trail',
 38: 'Gebel-barkal',
 39: 'Ghadames',
 40: 'Giza-pyramid-complex',
 41: 'Hatra',
 42: 'Hebron',
 43: 'Ichkeul-national-park',
 44: 'Islamic-cairo',
 45: 'Kadisha-valley',
 46: 'Kairouan',
 47: 'Kerkouane',
 48: 'Krak-des-chevaliers',
 49: 'Ksar-of-ait-ben-haddou',
 50: 'Leptis-magna',
 51: 'Medina-of-marrakesh',
 52: 'Medina-of-sousse',
 53: 'Medina-of-tunis',
 54: 'Meknes',
 55: 'Meroë',
 56: 'Necropolis-of-kerkouane',
 57: 'Bahrain-pearling-trail',
 58: 'Old-city-of-aleppo',
 59: 'Petra',
 60: 'Qalhat',
 61: 'Qasr-amra',
 62: 'Rabat',
 63: 'Rock-art-sites-of-tadrart-acacus',
 64: 'Sabratha',
 65: 'Samarra',
 66: 'Shibam',
 67: 'Site-of-palmyra',
 68: 'Theban-necropolis',
 69: 'Thebes,-egypt',
 70: 'Timgad',
 71: 'Tipaza',
 72: 'Tyre,-lebanon',
 73: 'Tétouan',
 74: 'Umm-ar-rasas',
 75: 'Volubilis',
 76: 'Wadi-al-hitan',
 77: 'Wadi-rum',
 78: 'Zabīd'
}

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
app();
