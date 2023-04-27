const tf = require('@tensorflow/tfjs');

const tensor = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
console.log(tensor.toString());