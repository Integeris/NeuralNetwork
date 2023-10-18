using System.Collections.Generic;

namespace NeuralNetwork
{
    /// <summary>
    /// Тренировочная задача.
    /// </summary>
    public class TrainingTask
    {
        /// <summary>
        /// Входные нейроны.
        /// </summary>
        private readonly List<Neuron> inputNeurons;

        /// <summary>
        /// Выходные нейроны.
        /// </summary>
        private readonly List<Neuron> outputNeurons;

        /// <summary>
        /// Входные нейроны.
        /// </summary>
        public List<Neuron> InputNeurons
        {
            get => inputNeurons;
        }

        /// <summary>
        /// Выходные нейроны.
        /// </summary>
        public List<Neuron> OutputNeurons
        {
            get => outputNeurons;
        }

        /// <summary>
        /// Создание тренировочной задачи.
        /// </summary>
        public TrainingTask()
        {
            inputNeurons = new List<Neuron>();
            outputNeurons = new List<Neuron>();
        }
    }
}
