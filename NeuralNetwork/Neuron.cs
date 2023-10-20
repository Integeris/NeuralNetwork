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
        /// Функция активации.
        /// </summary>
        private Func<float, float> activationFunction;

        /// <summary>
        /// Текущее значение.
        /// </summary>
        public float Value
        {
            get => value;
            set
            {
                this.value = value;
                activation = activationFunction(value);
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
        /// Функция активации.
        /// </summary>
        public Func<float, float> ActivationFunction
        {
            get => activationFunction;
            set => activationFunction = value;
        }

        /// <summary>
        /// Создание нейрона.
        /// </summary>
        public Neuron(Func<float, float>? activationFunction)
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException(nameof(activationFunction));
            }

            this.activationFunction = activationFunction;
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