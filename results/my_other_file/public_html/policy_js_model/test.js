const status = document.getElementById('status');
status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;

const MODEL_URL = './model.json';
const TEST_VALUE = [0,0,0,0,0,0,0,0,0,0]; #length10

async function run() {
    const model = await tf.loadLayersModel(MODEL_URL);

    // Print out the architecture of the loaded model.
    console.log(model.summary());

    // Create a 1 dimensional tensor with our test value.
    const input = tf.tensor1d(TEST_VALUE);

    // Actually make the prediction.
    const result = model.predict(input);

    // Grab the result of prediction using dataSync method
    // which ensures we do this synchronously.
    status.innerText = 'Input of ' + TEST_VALUE + 
        'sqft predicted as $' + result.dataSync()[0];
}

// Call our function to start the prediction!
run();
