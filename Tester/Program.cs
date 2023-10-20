using NeuralNetwork;
using System.Text;

namespace Tester
{
    public class Program
    {
        static void Main()
        {
            Console.Title = "Тестирование нейросети.";

            Network network = new Network(64, 64, 64, 32, 32);
            network.TrainSpeed = 2f;

            // Network network = new Network(2, 3, 1);

            //network.ActivationFunction = new Func<float, float>(x =>
            //{
            //    //if (x < 0)
            //    //{
            //    //    return 0;
            //    //}

            //    return x;
            //});

            //network.DerivativeActivationFunction = new Func<float, float>(x =>
            //{
            //    //if (x < 0)
            //    //{
            //    //    return 0;
            //    //}

            //    return 1;
            //});

            List<TrainingTask> trainingTasks = new List<TrainingTask>();

            //for (int i = 0; i < 2; i++)
            //{
            //    for (int j = 0; j < 2; j++)
            //    {
            //        TrainingTask trainingTask = new TrainingTask();
            //        trainingTask.InputNeurons.Add(new Neuron(network.ActivationFunction)
            //        {
            //            Value = i
            //        });

            //        trainingTask.InputNeurons.Add(new Neuron(network.ActivationFunction)
            //        {
            //            Value = j
            //        });

            //        trainingTask.OutputNeurons.Add(new Neuron(network.ActivationFunction)
            //        {
            //            Value = Convert.ToSingle(i != j)
            //        });

            //        trainingTasks.Add(trainingTask);
            //    }
            //}

            for (int i = 0; i <= 20; i++)
            {
                for (int j = 0; j <= 20; j++)
                {
                    TrainingTask task = new TrainingTask();

                    foreach (char chr in Convert.ToString(i, 2).PadLeft(32, '0'))
                    {
                        Neuron neuron = new Neuron(network.ActivationFunction)
                        {
                            Value = Convert.ToSingle(chr.ToString())
                        };

                        task.InputNeurons.Add(neuron);
                    }

                    foreach (char chr in Convert.ToString(j, 2).PadLeft(32, '0'))
                    {
                        Neuron neuron = new Neuron(network.ActivationFunction)
                        {
                            Value = Convert.ToSingle(chr.ToString())
                        };

                        task.InputNeurons.Add(neuron);
                    }

                    foreach (char chr in Convert.ToString(i + j, 2).PadLeft(32, '0'))
                    {
                        Neuron neuron = new Neuron(network.ActivationFunction)
                        {
                            Value = Convert.ToSingle(chr.ToString())
                        };

                        task.OutputNeurons.Add(neuron);
                    }

                    trainingTasks.Add(task);
                }
            }

            //for (int i = 0; i < 50; i++)
            //{
            //    for (int j = 0; j < 50; j++)
            //    {
            //        TrainingTask trainingTask = new TrainingTask();
            //        trainingTask.InputNeurons.Add(new Neuron(network.ActivationFunction)
            //        {
            //            Value = i
            //        });
            //        trainingTask.InputNeurons.Add(new Neuron(network.ActivationFunction)
            //        {
            //            Value = j
            //        });
            //        trainingTask.OutputNeurons.Add(new Neuron(network.ActivationFunction)
            //        {
            //            Value = i + j
            //        });
            //    }
            //}

            Console.WriteLine("До обучения");

            for (int i = 0; i <= 20; i += 5)
            {
                for (int j = 0; j <= 20; j += 5)
                {
                    string tmp = Convert.ToString(i, 2).PadLeft(32, '0');

                    for (int k = 0; k < 32; k++)
                    {
                        network.InputLayer[k].Value = Convert.ToSingle(tmp[k].ToString());
                    }

                    tmp = Convert.ToString(j, 2).PadLeft(32, '0');

                    for (int k = 32; k < 64; k++)
                    {
                        network.InputLayer[k].Value = Convert.ToSingle(tmp[k % 32].ToString());
                    }

                    network.Execute();

                    PrintNetwork(network, i + j);
                }
            }

            //for (int i = 0; i < 2; i++)
            //{
            //    for (int j = 0; j < 2; j++)
            //    {
            //        network.InputLayer[0].Value = i;
            //        network.InputLayer[1].Value = j;

            //        network.Execute();

            //        PrintNetwork(network, Convert.ToSingle(i != j));
            //    }
            //}

            for (int i = 0; i < 100; i++)
            {
                network.Train(trainingTasks);
                Console.WriteLine("Проход {0,5}) MSE {1:p}", i, network.MSE);
            }

            Console.WriteLine("\nПосле обучения");

            //for (int i = 0; i < 2; i++)
            //{
            //    for (int j = 0; j < 2; j++)
            //    {
            //        network.InputLayer[0].Value = i;
            //        network.InputLayer[1].Value = j;

            //        network.Execute();

            //        PrintNetwork(network, Convert.ToSingle(i != j));
            //    }
            //}

            for (int i = 0; i <= 20; i += 5)
            {
                for (int j = 0; j <= 20; j += 5)
                {
                    string tmp = Convert.ToString(i, 2).PadLeft(32, '0');

                    for (int k = 0; k < 32; k++)
                    {
                        network.InputLayer[k].Value = Convert.ToSingle(tmp[k].ToString());
                    }

                    tmp = Convert.ToString(j, 2).PadLeft(32, '0');

                    for (int k = 32; k < 64; k++)
                    {
                        network.InputLayer[k].Value = Convert.ToSingle(tmp[k % 32].ToString());
                    }

                    network.Execute();

                    PrintNetwork(network, i + j);
                }
            }
        }

        private static void PrintNetwork(Network network, float answer)
        {
            //StringBuilder stringBuilder = new StringBuilder();
            //stringBuilder.Append("Входные данные:  ");
            //stringBuilder.AppendLine(String.Join<Neuron>("; ", network.InputLayer));
            //stringBuilder.Append("Выходные данные: ");
            //stringBuilder.Append(String.Join<Neuron>("; ", network.OutputLayer));
            //stringBuilder.Append($"MSE: {network.OutputLayer.Sum(i => Math.Pow(answer - i.Activation, 2)) / network.OutputLayer.Length:p}");

            string tmp = Convert.ToString((int)answer, 2).PadLeft(32, '0');
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append("Входные данные:  ");
            // stringBuilder.AppendLine(String.Join<Neuron>("; ", network.InputLayer));
            stringBuilder.AppendLine($"{String.Join("", network.InputLayer.Take(32).Select(i => i.Value))} ({GetNum(network.InputLayer.Take(32).Select(i => i.Value).ToArray())})");
            stringBuilder.AppendLine($"\t\t {String.Join("", network.InputLayer.Skip(32).Select(i => i.Value))} ({GetNum(network.InputLayer.Skip(32).Select(i => i.Value).ToArray())})");
            stringBuilder.Append("Нужный результат:");
            stringBuilder.AppendLine($"{String.Join("", tmp.Select(i => i))} ({(int)answer})");
            stringBuilder.Append("Выходные данные: ");
            stringBuilder.AppendLine($"{String.Join("", network.OutputLayer.Select(i => Math.Round(i.Activation)))} ({GetNum(network.OutputLayer.Select(i => i.Activation).ToArray())})");

            double mse = 0;

            for (int i = 0; i < network.OutputLayer.Length; i++)
            {
                mse += Math.Pow(Single.Parse(tmp[i].ToString()) - network.OutputLayer[i].Activation, 2);
            }

            stringBuilder.Append($"MSE: {mse / network.OutputLayer.Length:p}");

            Console.WriteLine(stringBuilder.ToString());
        }

        private static int GetNum(float[] floats)
        {
            int result = 0;

            for (int i = 0; i < floats.Length; i++)
            {
                result |= (int)Math.Round(floats[i], 0) << floats.Length - i - 1;
            }

            return result;
        }
    }
}