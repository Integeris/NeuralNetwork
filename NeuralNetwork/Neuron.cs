using System;

namespace NeuralNetwork
{
    /// <summary>
    /// Нейрон.
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Текущее значение.
        /// </summary>
        private float value;

        /// <summary>
        /// Активация.
        /// </summary>
        private float activation;

        /// <summary>
        /// Доля от ошибки.
        /// </summary>
        private float delta;

        /// <summary>
        /// Текущее значение.
        /// </summary>
        public float Value
        {
            get => value;
            set
            {
                this.value = value;
                activation = (float)(1 / (1 + Math.Exp(-value)));
            }
        }

        /// <summary>
        /// Активация.
        /// </summary>
        public float Activation
        {
            get => activation;
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
        /// Создание нейрона.
        /// </summary>
        public Neuron() 
        {
            Value = 0;
        }

        /// <summary>
        /// Вывод значения нейрона.
        /// </summary>
        /// <returns>Представление значения нейрона.</returns>
        public override string ToString()
        {
            return $"{activation:f4}({value:f3})";
        }
    }
}