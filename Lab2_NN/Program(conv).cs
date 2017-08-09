using System;
using System.Collections.Generic;

namespace Neuronets
{
    class Program
    {
        static void Main(string[] args)
        {
            //Конвертер валют
            var nnConv = new Neuronet(new[] {1, 2, 1}, FunctionType.Linear);
                //конструктор, задающий общую структуру сети
            foreach (var layer in nnConv.Layers)
                layer.FuncDefaultParam = 1.0; //задание параметра активационной функции 

            nnConv.SetLinks();

            var np = new NeuroPreceptor(nnConv, 0.000000001);
            //1,113 | 0,898
            //обучающая выборка
            var r = RandomProvider.GetThreadRandom();
            var examples = new List<double[]>();
            for (int i = 0; i < 20; ++i)
                examples.Add(new double[] {r.Next(0, 100)});
            
            //требуемые значения
            double[][] answers = new double[examples.Count][];
            for (int i = 0; i < examples.Count; ++i)
                answers[i] = new [] {examples[i][0] * 1.113};
            //Обучение
            double err = 30;
            int epoch = 1;
            int n = 1;
            while (err > 0.0005)
            {
                err = np.SupervisedTraining(SupervisedMethod.GradientDescent, examples.ToArray(), answers);
                Console.WriteLine("Эпоха: " + epoch + "    Ошибка сети: " + err);
                //Console.WriteLine("Ответ сети: " + nnConv.Activate(new double[] {1})[0]);
                ++epoch;
                if (epoch == 100000 * n)
                {
                    ++n;
                    Console.WriteLine("Желаете продолжить? (y/n)");
                    if (Console.Read() == 'n')
                        break;
                }
            }
            np.SaveExpToFile("Convert");
            Console.WriteLine("Обучение закончено! Коэффициенты сохранены в файл.");

            //np.LoadExpFromFile("Convert");
            //Console.WriteLine("Коэффициенты загружены.");

            double[] test = new double[1];
            while (true)
            {
                Console.WriteLine("\nВведите сумму в $:");
                if (!Double.TryParse(Console.ReadLine(), out test[0]))
                    continue;
                Console.WriteLine("Ответ сети: " + nnConv.Activate(test)[0]);
                Console.WriteLine("Контрольное значение: " + test[0] * 1.113);
            }
        } 
    }
}
