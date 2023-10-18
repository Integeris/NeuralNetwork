using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    /// <summary>
    /// Нейронная сеть.
    /// </summary>
    public class Network
    {
        /// <summary>
        /// Ошибка.
        /// </summary>
        private float mse;

        /// <summary>
        /// Скорость обучения.
        /// </summary>
        private float trainSpeed;

        /// <summary>
        /// Момент.
        /// </summary>
        private float moment;

        /// <summary>
        /// Слои нейронов.
        /// </summary>
        private readonly List<Neuron[]> neuronLayers;

        /// <summary>
        /// Слои весов.
        /// </summary>
        private readonly List<Weight[]> weightLayers;

        /// <summary>
        /// Ошибка.
        /// </summary>
        public float MSE
        {
            get => mse;
        }

        /// <summary>
        /// Скорость обучения.
        /// </summary>
        public float TrainSpeed
        {
            get => trainSpeed;
            set => trainSpeed = value;
        }

        /// <summary>
        /// Момент.
        /// </summary>
        public float Moment
        {
            get => moment;
            set => moment = value;
        }

        /// <summary>
        /// Входной слой.
        /// </summary>
        public Neuron[] InputLayer
        {
            get => neuronLayers[0];
        }

        /// <summary>
        /// Выходной слой.
        /// </summary>
        public Neuron[] OutputLayer
        {
            get => neuronLayers[^1];
        }

        /// <summary>
        /// Создание нейронной сети.
        /// </summary>
        /// <param name="layersNeuronsCount">Количество нейронов в слоях.</param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Network(params int[] layersNeuronsCount)
        {
            if (layersNeuronsCount.Length < 2)
            {
                throw new ArgumentOutOfRangeException(nameof(layersNeuronsCount), "Необходимо как минимум два слоя.");
            }

            trainSpeed = 0.7f;
            moment = 0.3f;

            neuronLayers = new List<Neuron[]>();
            weightLayers = new List<Weight[]>();

            foreach (int count in layersNeuronsCount)
            {
                Neuron[] layer = new Neuron[count];

                for (int i = 0; i < count; i++)
                {
                    layer[i] = new Neuron();
                }

                neuronLayers.Add(layer);
            }

            int weightesLayersCount = layersNeuronsCount.Length - 1;

            for (int currentLayer = 0; currentLayer < weightesLayersCount; currentLayer++)
            {
                int weightCount = 0;
                int nextLayer = currentLayer + 1;
                Weight[] layer = new Weight[neuronLayers[currentLayer].Length * neuronLayers[nextLayer].Length];

                foreach (Neuron inputNeuron in neuronLayers[currentLayer])
                {
                    foreach (Neuron outputNeuron in neuronLayers[nextLayer])
                    {
                        Weight weight = new Weight(inputNeuron, outputNeuron)
                        {
                            Value = new Random().Next(-30000, 30000) / 10000
                        };

                        layer[weightCount] = weight;
                        weightCount++;
                    }
                }

                weightLayers.Add(layer);
            }
        }

        /// <summary>
        /// Выполнить.
        /// </summary>
        public void Execute()
        {
            for (int i = 1; i < neuronLayers.Count; i++)
            {
                for (int j = 0; j < neuronLayers[i].Length; j++)
                {
                    IEnumerable<Weight> weights = weightLayers[i - 1].Where(weight => weight.OutputNeuron == neuronLayers[i][j]);
                    float sum = 0;

                    foreach (Weight weight in weights)
                    {
                        sum += (float)weight.Execute();
                    }

                    neuronLayers[i][j].Value = sum;
                }
            }
        }

        /// <summary>
        /// Тренировка нейронной сети.
        /// </summary>
        /// <param name="trainingTasks">Тренировочные задачи.</param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public void Train(List<TrainingTask> trainingTasks)
        {
            if (trainingTasks.Any(task => InputLayer.Length != task.InputNeurons.Count))
            {
                throw new ArgumentOutOfRangeException(nameof(trainingTasks), "Количество входных нейронов тренировочных задач должно совпадать с количеством входных нейронов нейронной сети.");
            }
            else if (trainingTasks.Any(task => OutputLayer.Length != task.OutputNeurons.Count))
            {
                throw new ArgumentOutOfRangeException(nameof(trainingTasks), "Количество выходных нейронов тренировочных задач должно совпадать с количеством выходных нейронов нейронной сети.");
            }

            foreach (TrainingTask task in trainingTasks)
            {
                for (int i = 0; i < InputLayer.Length; i++)
                {
                    InputLayer[i].Value = task.InputNeurons[i].Value;
                }

                Execute();

                {
                    float mse = 0;

                    for (int i = 0; i < OutputLayer.Length; i++)
                    {
                        float tmp = task.OutputNeurons[i].Value - OutputLayer[i].Activation;
                        mse += (float)Math.Pow(tmp, 2);
                        OutputLayer[i].Delta = (1 - OutputLayer[i].Activation) * OutputLayer[i].Activation * tmp;
                    }

                    mse /= OutputLayer.Length;
                    this.mse += mse;
                }

                for (int i = neuronLayers.Count - 2; i >= 0; i--)
                {
                    for (int j = 0; j < neuronLayers[i].Length; j++)
                    {
                        Neuron curentNeuron = neuronLayers[i][j];
                        Weight[] weights = weightLayers[i].Where(weight => weight.InputNeuron == curentNeuron).ToArray();
                        curentNeuron.Delta = 0;

                        foreach (Weight weight in weights)
                        {
                            curentNeuron.Delta += weight.Value * weight.OutputNeuron.Delta;
                        }

                        curentNeuron.Delta *= (1 - curentNeuron.Activation) * curentNeuron.Activation;

                        foreach (Weight weight in weights)
                        {
                            float grad = curentNeuron.Activation * weight.OutputNeuron.Delta;
                            weight.Delta = trainSpeed * grad + moment * weight.Delta;
                            weight.Value += weight.Delta;
                        }
                    }
                }
            }

            mse /= trainingTasks.Count;
        }
    }
}
