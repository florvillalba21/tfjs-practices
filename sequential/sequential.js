const inpEpoca = document.getElementById('inpEpoca');
const inpX = document.getElementById('inpX');

const model = tf.sequential();

//funciona para graficar las perdidas
const plotLoss = (losses) => {
  const values = losses.map((loss, i) => ({ x: i + 1, y: loss }));


  //se almacena en un objeto los valores recaudados anteriormente
  //los valores serán de la variable values y la etiqueta es perdidas
  const series = { values: [values], series: ['Perdidas'] };

  //renderiza el grafico de lineas, con nombre perdidas
  tfvis.render.linechart({ name: 'perdidas' }, series);
};

const fit = async() => {
  const epocas = parseInt(inpEpoca.value);

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  const xs = tf.tensor2d([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 10], [11, 1]);
  const ys = tf.tensor2d([-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 19], [11, 1]);

  //variable para almacenar las perdidas
  const losses = [];

  await model.fit(xs, ys, {
    epochs: epocas,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        //se agregan las perdidas al arreglo
        losses.push(logs.loss);
        console.log(logs);
        console.log('\n');
        console.log(`Epoch : ${epoch + 1} - Loss: ${logs.loss.toFixed(4)}`);
        plotLoss(losses);
      },
    },
  });

  document.getElementById('res').innerText = 'Entrenamiento terminado';
};

const predict = () => {
  const inp = parseInt(inpX.value);
  document.getElementById('micro-out-div').innerText = model
    .predict(tf.tensor2d([inp], [1, 1]))
    .dataSync();
};
