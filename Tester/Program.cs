using NeuralNetwork;
using System.Text;

namespace Tester
{
    public class Program
    {
        static void Main()
        {
            Console.Title = "Тестирование нейросети.";

            Network network = new Network(2, 2, 1);
            List<TrainingTask> trainingTasks = new List<TrainingTask>();

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    TrainingTask task = new TrainingTask()
                    {
                        InputNeurons =
                        {
                            new Neuron()
                            {
                                Value = i
                            },
                            new Neuron()
                            {
                                Value = j
                            }
                        },
                        OutputNeurons =
                        {
                            new Neuron()
                            {
                                Value = Convert.ToSingle(i != j)
                            }
                        }
                    };

                    trainingTasks.Add(task);
                }
            }

            Console.WriteLine("До обучения");

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    network.InputLayer[0].Value = i;
                    network.InputLayer[1].Value = j;

                    network.Execute();

                    PrintNetwork(network);
                }
            }

            for (int i = 0; i < 900000; i++)
            {
                network.Train(trainingTasks);
            }

            Console.WriteLine("\nПосле обучения");

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    network.InputLayer[0].Value = i;
                    network.InputLayer[1].Value = j;

                    network.Execute();

                    PrintNetwork(network);
                }
            }
        }

        private static void PrintNetwork(Network network)
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append("Входные данные:  ");
            stringBuilder.AppendLine(String.Join<Neuron>("; ", network.InputLayer));
            stringBuilder.Append("Выходные данные: ");
            stringBuilder.Append(String.Join<Neuron>("; ", network.OutputLayer));

            Console.WriteLine(stringBuilder.ToString());
        }
    }
}