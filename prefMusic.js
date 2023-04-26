const tf = require("@tensorflow/tfjs");

const usuarios = ["Gant", "Todd", "Jed", "Justin"];
const bandas = [
  "Nirvana",
  "Uñas de nueve pulgadas",
  "Backstreet Boys",
  "N sincronización",
  "Club nocturno",
  "apashe",
  "PLS",
];

const caracteristicas = [
  "grunge",
  "rock",
  "industrial",
  "banda de chicos",
  "danza",
  "tecno",
];

const voto_usuario = tf.tensor([
  [10, 9, 1, 1, 8, 7, 8],
  [6, 8, 2, 2, 0, 10, 0],
  [0, 2, 10, 9, 3, 7, 0],
  [7, 4, 2, 3, 6, 5, 5],
]);

const banda_feats = tf.tensor([
  [1, 1, 0, 0, 0, 0],
  [1, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1],
  [0, 0, 1, 0, 0, 1],
  [1, 1, 0, 0, 0, 0],
]);

const user_feats = tf.matMul(voto_usuario, banda_feats)

user_feats.print()