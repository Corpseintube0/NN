using System;
using System.Collections.Generic;

namespace Neuronets
{
    class Program
    {
        static void Main(string[] args)
        {
            //test
            //Умножение 3-х чисел [0;1]
            var myltiply3 = new Neuronet(new[] {3, 3, 2, 1}, FunctionType.Sigmoid); //конструктор, задающий общую структуру сети
            foreach (var layer in myltiply3.Layers)
                layer.FuncDefaultParam = 1.0;         //задание параметра активационной функции 

            myltiply3.SetLinks(); //выставление связей
            //myltiply3.SetLinks(0, new[,] {{true, false}, {true, false},{false, true} });

            var np = new NeuroPreceptor(myltiply3, 0.85); //обучающий класс
            
            //обучающая выборка
            var r = RandomProvider.GetThreadRandom();
            var examples = new List<double[]>
                               {
                                   new double[] {1, 1, 1},
                                   new double[] {0, r.NextDouble(), r.NextDouble()}, 
                                   new double[] {r.NextDouble(), 0, r.NextDouble()}, 
                                   new double[] {r.NextDouble(), r.NextDouble(), 0},
                                   new double[] {1, r.NextDouble(), 1}, 
                                   new double[] {1, 1, r.NextDouble()},
                                   new double[] {r.NextDouble(), 1, 1},
                                   new double[] {r.NextDouble(), r.NextDouble() , 1},
                                   new double[] {1, r.NextDouble() , r.NextDouble()},
                                   new double[] {r.NextDouble(), 1 , r.NextDouble()},
                               };
            for (int i = 0; i < 64; ++i)
                examples.Add(new double[] {r.NextDouble(), r.NextDouble(), r.NextDouble()});
            
            //требуемые значения
            double[][] answers = new double[examples.Count][];
            for (int i = 0; i < examples.Count; ++i) //1 / (1 + e ^ (x1*x2*x3) )
                answers[i] = new[] {1/(1+Math.Exp(examples[i][0] * examples[i][1] * examples[i][2]))};
            
            //Обучение
            //np.LoadExpFromFile("myltiply");
            double err = 20;
            UInt64 epoch = 1;
            ulong n = 1;
            while(err > 0.0005 && Math.Abs(err - 0.0005) > 0.00001)
            {
                err = np.SupervisedTraining(SupervisedMethod.GradientDescent, examples.ToArray(), answers);
                Console.WriteLine("Эпоха: " + epoch + "    Ошибка сети: " + err);
                ++epoch;
                if (epoch == 100000*n)
                {
                    ++n;
                    Console.WriteLine("Желаете продолжить? (y/n)");
                    if (Console.Read() == 'n')
                        break;
                }
            }
            np.SaveExpToFile("myltiply");
            Console.WriteLine("Обучение закончено! Коэффициенты сохранены в файл.");
            
            //np.LoadExpFromFile("myltiply");
            //Console.WriteLine("Коэффициенты загружены.");
            
            double[] test = new double[3];
            while (true)
            {
                Console.WriteLine("\nВведите 3 числа:");
                for (int i = 0; i < 3; ++i)
                {
                    Console.Write("x" + (i+1) + ": ");
                    if (!Double.TryParse(Console.ReadLine(), out test[i]))
                        --i;
                }
                Console.Write("Результат: " + Math.Log(1/myltiply3.Activate(test)[0] - 1) + "\n"); //ln(1/x - 1)
            }
            
            //endtest
        } 
    }
}
