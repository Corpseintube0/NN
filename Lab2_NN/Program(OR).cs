using System;

namespace Neuronets
{
    class Program
    {
        static void Main(string[] args)
        {
            //test
            //Операция "или"
            var nnOr = new Neuronet(new[] {2, 2, 2, 1}, FunctionType.Sigmoid); //конструктор, задающий общую структуру сети
            foreach (var layer in nnOr.Layers)
                layer.FuncDefaultParam = 1;         //задание параметра активационной функции 

            nnOr.SetLinks();
            nnOr.SetLinks(0, new[,] { { true, true }, { true, true } }); //формирование связей
            nnOr.SetLinks(1, new[,] { { true, true }, { true, true } });
            nnOr.SetLinks(2, new[,] { { true }, { true } });
            nnOr.SetLinks(3, new[,] { { true } });

            //nnOr[0][0].SetInputsNumber(1);
            //nnOr[0][1].SetInputsNumber(1);
            //nnOr[0][0].Weights[0] = 1.0;
            //nnOr[0][1].Weights[0] = 1.0;
            //nnOr[0][0].Offset = 0.0;
            //nnOr[0][1].Offset = 0.0;

            var np = new NeuroPreceptor(nnOr, 1);
            //обучающая выборка
            double[][] examples = new[]
                                      {
                                          new []{0.0, 0.0}, 
                                          new []{0.0, 1.0}, 
                                          new []{1.0, 0.0}, 
                                          new []{1.0, 1.0}
                                      };
            //требуемые значения
            double[][] answers = new[]
                                     {
                                         new[] {0.0},
                                         new[] {1.0},
                                         new[] {1.0},
                                         new[] {1.0}
                                     };
            //Обучение
            double err = 1;
            int epoch = 1;
            while(err > 0.0001)
            {
                err = np.SupervisedTraining(SupervisedMethod.GradientDescent, examples, answers);
                Console.WriteLine("Эпоха: " + epoch + "    Ошибка сети: " + err);
                ++epoch;
            }
            np.SaveExpToFile("LogSigmoidOr");
            Console.WriteLine("Обучение закончено! Коэффициенты сохранены в файл.");

            //np.LoadExpFromFile("LogSigmoidOr");
            //Console.WriteLine("Коэффициенты загружены.");

            Console.ReadKey(true);
            double[][] ans = new double[4][];
            ans[0] = nnOr.Activate(new[] { 0.0, 0.0 });
            ans[1] = nnOr.Activate(new[] { 0.0, 1.0 });
            ans[2] = nnOr.Activate(new[] { 1.0, 0.0 });
            ans[3] = nnOr.Activate(new[] { 1.0, 1.0 });

            for (int i = 0; i<4; ++i)
            {
                Console.WriteLine(ans[i][0]);
            }
            Console.ReadKey();

            //endtest
        } 
    }
}
