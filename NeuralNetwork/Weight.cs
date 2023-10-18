namespace NeuralNetwork
{
    /// <summary>
    /// Вес.
    /// </summary>
    public class Weight
    {
        /// <summary>
        /// Множитель.
        /// </summary>
        private float value;

        /// <summary>
        /// Доля от ошибки.
        /// </summary>
        private float delta;

        /// <summary>
        /// Входной нейрон.
        /// </summary>
        private Neuron inputNeuron;

        /// <summary>
        /// Выходной нейрон.
        /// </summary>
        private Neuron outputNeuron;

        /// <summary>
        /// Множитель.
        /// </summary>
        public float Value
        {
            get => value;
            set => this.value = value;
        }

        /// <summary>
        /// Доля от ошибки.
        /// </summary>
        public float Delta
        {
            get => delta;
            set => delta = value;
        }

        /// <summary>
        /// Входной нейрон.
        /// </summary>
        public Neuron InputNeuron
        {
            get => inputNeuron;
            set => inputNeuron = value;
        }

        /// <summary>
        /// Выходной нейрон.
        /// </summary>
        public Neuron OutputNeuron
        {
            get => outputNeuron;
            set => outputNeuron = value;
        }

        /// <summary>
        /// Создание веса.
        /// </summary>
        /// <param name="inputNeuron">Входной нейрон.</param>
        /// <param name="outputNeuron">Выходной нейрон.</param>
        public Weight(Neuron inputNeuron, Neuron outputNeuron) 
        {
            this.inputNeuron = inputNeuron;
            this.outputNeuron = outputNeuron;
        }

        /// <summary>
        /// Посчитать значение весов.
        /// </summary>
        /// <returns>Результат выполнения весов.</returns>
        public float Execute()
        {
            return inputNeuron.Activation * value;
        }
    }
}
