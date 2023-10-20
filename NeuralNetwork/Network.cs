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
        private double mse;

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
        /// Функция активации.
        /// </summary>
        private Func<float, float> activationFunction;

        /// <summary>
        /// Производная функции активации.
        /// </summary>
        private Func<float, float> derivativeActivationFunction;

        /// <summary>
        /// Ошибка.
        /// </summary>
        public double MSE
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
        /// Функция активации.
        /// </summary>
        public Func<float, float> ActivationFunction
        {
            get => activationFunction;
            set
            {
                activationFunction = value;

                foreach (Neuron[] layer in neuronLayers)
                {
                    foreach (Neuron neuron in layer)
                    {
                        neuron.ActivationFunction = value;
                    }
                }
            }
        }

        /// <summary>
        /// Производная функции активации.
        /// </summary>
        public Func<float, float> DerivativeActivationFunction
        {
            get => derivativeActivationFunction;
            set => derivativeActivationFunction = value;
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

            activationFunction = new Func<float, float>(value => (float)(1 / (1 + Math.Exp(-value))));
            derivativeActivationFunction = new Func<float, float>((value) => (1 - value) * value);

            trainSpeed = 0.7f;
            moment = 0.3f;

            neuronLayers = new List<Neuron[]>();
            weightLayers = new List<Weight[]>();

            foreach (int count in layersNeuronsCount)
            {
                Neuron[] layer = new Neuron[count];

                for (int i = 0; i < count; i++)
                {
                    layer[i] = new Neuron(activationFunction);
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
                            Value = (float)new Random().Next(-3000, 3000) / 1000
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
                    double mse = 0;

                    for (int i = 0; i < OutputLayer.Length; i++)
                    {
                        float tmp = task.OutputNeurons[i].Value - OutputLayer[i].Activation;
                        mse += Math.Pow(tmp, 2);
                        OutputLayer[i].Delta = derivativeActivationFunction(OutputLayer[i].Activation) * tmp;
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

                        curentNeuron.Delta *= derivativeActivationFunction(curentNeuron.Activation);

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
