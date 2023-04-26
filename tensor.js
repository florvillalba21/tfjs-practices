const tf = require("@tensorflow/tfjs");

console.log("start", tf.memory().numTensors);

let keeper, chaser, seeker, beater;

tf.tidy(() => {
  keeper = tf.tensor([1, 2, 3]);
  chaser = tf.tensor([1, 2, 3]);
  seeker = tf.tensor([1, 2, 3]);
  beater = tf.tensor([1, 2, 3]);


  console.log("dentro ordenado", tf.memory().numTensors);

  tf.keep(keeper); 
  return chaser;
});

console.log("después de ordenar", tf.memory().numTensors);

keeper.dispose();
chaser.dispose();

console.log("end", tf.memory().numTensors);
