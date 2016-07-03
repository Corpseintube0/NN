using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;


namespace Neuronets
{
    /// <summary>
    /// Класс, предоставляющий методы обучения нейронной сети.
    /// </summary>
    public class NeuroPreceptor
    {
        #region  Поля

        /// <summary>
        /// Обучаемая нейронная сеть.
        /// </summary>
        private Neuronet _net;

        /// <summary>
        /// Скорость обучения. Диапазон допустимых значений: (0 ;1]
        /// </summary>
        private double _ts;

        /// <summary>
        /// Делегат для инкапсуляции метода обучения с учителем.
        /// </summary>
        private SupervisedDelegate _sd;

        #endregion

        #region Свойства

        /// <summary>
        /// Получает или задает скорость обучения.
        /// </summary>
        public double TrainingSpeed
        {
            get
            {
                return _ts;
            }
            set
            {
                if (value > 1)
                    _ts = 1;
                else
                    if (value < 0.0000001)
                        _ts = 0.0000001;
                    else
                        _ts = value;
            }
        }

        #endregion

        #region  Делегаты

        /// <summary>
        /// Делегат для обучения с учителем.
        /// </summary>
        private delegate double[][] SupervisedDelegate(double[] input, double[] required);

        #endregion

        #region Конструкторы

        /// <summary>
        /// Инициализирует новый экземпляр класса NeuroPreceptor.
        /// </summary>
        /// <param name="student">Обучаемая нейронная сеть.</param>
        /// <param name="trainingSpeed">Скорость обучения.</param>
        public NeuroPreceptor(Neuronet student, double trainingSpeed = 0.8)
        {
            _net = student;
            TrainingSpeed = trainingSpeed;
        }

        #endregion

        #region Методы

        /// <summary>
        /// Одна эпоха обучения сети методом обучения с учителем.
        /// </summary>
        /// <param name="method">Метод обучения.</param>
        /// <param name="inputSets">Входной набор обучающей выборки.</param>
        /// <param name="requiredValues">Образец требуемых значений.</param>
        /// <returns>Ошибка сети после одной эпохи обучения.</returns>
        public double SupervisedTraining(SupervisedMethod method, double[][] inputSets, double[][] requiredValues)
        {
            //TODO: сделать проверку корректности введенных данных 2) подходят ли эти наборы к сети
            if (inputSets.GetLength(0) != requiredValues.GetLength(0))
                throw new Exception("Несоответствие размеров наборов обучающей выборки");

            switch (method)
            {
                case SupervisedMethod.GradientDescent:
                    _sd = GradientDescent;
                    break;
                default:
                    throw new Exception("Unknown method: " + method);
            }
            for (int i = 0; i<inputSets.GetLength(0); ++i)
            {
                var errors = _sd(inputSets[i], requiredValues[i]);
                CorrectWeights(errors);
            }
            double netError = 0;
            for (int i = 0; i<inputSets.GetLength(0); ++i)
            {
                for (int j = 0; j < requiredValues[0].Length; ++j)
                    netError += Math.Pow((_net.Activate(inputSets[i])[j] - requiredValues[i][j]), 2);
            }
            return netError / 2.0;
        }

        /// <summary>
        /// Корректирует веса на основе значений ошибок нейронов.
        /// </summary>
        /// <param name="neuronErrs">Значения ошибки нейронов выходного и скрытых слоёв.</param>
        private void CorrectWeights(double[][] neuronErrs)
        {
            //for (int i=1; i < _net.LayersCount; ++i) //i=1 индекс первого скрытого слоя
            for (int i = _net.LayersCount - 1; i > 0; --i)
            {
                if (neuronErrs[i - 1].Length != _net[i].NeuronsCount)
                    throw new Exception("Число нейронов (слой " + i + "): " + _net[i].NeuronsCount +
                                        ". Элементов массива: " + neuronErrs[i - 1].Length + '.');
                for (int j=0; j < _net[i].NeuronsCount; ++j)
                {
                    for (int k=0; k < _net[i][j].Weights.Length; ++k)
                    {
                        //вычисляем от какого нейрона идет связь к данному нейрону
                        int linkInd = -1;

                        int l;
                        for (l = 0; l < _net[i-1].NeuronsCount; ++l)
                        {
                            if (_net[i - 1].Links[l, j])
                                ++linkInd;
                            if (linkInd == k)
                            {
                                _net[i][j].Weights[k] += -_ts * neuronErrs[i - 1][j] * _net[i - 1][l].SignalOut(); //корректируем вес
                                break;
                            }
                        }
                        _net[i][j].Offset += -_ts * neuronErrs[i - 1][j]; //корректируем смещение
                    }
                }
            }
        }

        /// <summary>
        /// Метод градиентного спуска для обучающего алгоритма обратного распространения ошибки.
        /// </summary>
        /// <param name="inputSet">Входные данные из обучающей выборки.</param>
        /// <param name="requiredValues">Требуемые выходные значения.</param>
        /// <returns>Значения ошибки нейронов выходного и скрытых слоёв.</returns>
        private double[][] GradientDescent(double[] inputSet, double[] requiredValues)
        {
            var factValues = _net.Activate(inputSet); //значения для вычисления ошибки сети
            var lastLayerErr = new double[requiredValues.Length];
            for (int i=0; i < lastLayerErr.Length; ++i)
            {
                //вычисляем ошибку выходного слоя
                switch (_net[i].LayerFunction)
                {
                    case FunctionType.Sigmoid:
                        lastLayerErr[i] = _net.OutputLayer.FuncDefaultParam * factValues[i] * (1 - factValues[i]) *
                                          (factValues[i] - requiredValues[i]);
                        break;
                    case FunctionType.Linear:
                        lastLayerErr[i] = _net.OutputLayer.FuncDefaultParam * (factValues[i] - requiredValues[i]);
                        break;
                    default:
                        throw new Exception(_net[i].LayerFunction + " is not done yet... Sorry.");
                }
            }
            var layersErr = new List<double[]> {lastLayerErr};
            int hiddenLayersCount = _net.LayersCount - 2; //число скрытых слоёв
            for (int i=hiddenLayersCount; i > 0; --i)
            {
                var hLayerErr = new double[_net[i].NeuronsCount]; //массив с ошибками i-го скрытого слоя
                double sumWeightsErr = 0;   //сумма весов нейронов i+1 слоя, соединенных с j-м нейроном i-го слоя  
                for (int j = 0; j < _net[i].NeuronsCount; ++j)
                {
                    //узнаем с какими нейронами соединен j-й нейрон
                    for (int k = 0; k < _net[i + 1].NeuronsCount; ++k)
                    {
                        int x = -1; //x - индекс веса
                        if (_net[i].Links[j, k]) //если j-й нейрон соединен с k-м нейроном
                        {
                            ++x;
                            sumWeightsErr += layersErr[layersErr.Count - 1][k] * _net[i + 1][k].Weights[x];
                        }
                    }
                    //вычисляем ошибку скрытого слоя
                    switch (_net[i].LayerFunction)
                    {
                        case FunctionType.Sigmoid:
                            hLayerErr[j] = sumWeightsErr * _net[i].FuncDefaultParam * _net[i][j].SignalOut() *
                                           (1 - _net[i][j].SignalOut());
                            break;
                        case FunctionType.Linear:
                            hLayerErr[j] = sumWeightsErr * _net[i].FuncDefaultParam;
                            break;
                        default:
                            throw new Exception(_net[i].LayerFunction + " is not done yet... Sorry.");
                    }
                }
                layersErr.Add(hLayerErr);
            }
            layersErr.Reverse();
            return layersErr.ToArray();
        }
        

        /// <summary>
        /// Загружает весовые коэффициенты нейросети из файла.
        /// </summary>
        /// <param name="fileName">Имя файла с коэффициентами.</param>
        public void LoadExpFromFile(String fileName)
        {
            var fi = new FileInfo(fileName);
            var fs = fi.Open(FileMode.Open, FileAccess.Read);
            var br = new BinaryReader(fs);

            int layersCount = br.ReadInt32(); //считали первые 4 байта (число слоёв)
            if (layersCount != _net.LayersCount)
                throw new Exception("Число слоёв в считываемом файле \" " + fileName + "\" не совпадает с числом слоёв сети.");
            var neuronsCount = new Int32[layersCount];
            for (int i=0; i < layersCount; ++i)
            {
                neuronsCount[i] = br.ReadInt32(); //считали число нейронов в i-ом слое
                if (neuronsCount[i] != _net[i].NeuronsCount)
                    throw new Exception("Число нейронов слоя #" + i.ToString(CultureInfo.InvariantCulture) +
                                        " не совпадает с числом нейронов в считываемом файле \"" + fileName + '"');
            }
            for (int i = 0; i < layersCount; ++i)
            {
                for (int j = 0; j < neuronsCount[i]; ++j)
                {
                    _net[i][j].Offset = br.ReadDouble(); //считали смещение
                    for (int k = 0; k < _net[i][j].Weights.Length; ++k)
                        _net[i][j].Weights[k] = br.ReadDouble(); //считали вес #k для нейрона #j для слоя #i 
                }
            }
            br.Dispose();
            br.Close();
        }

        /// <summary>
        /// Сохраняет текущие весовые коэффициенты нейросети в файл.
        /// </summary>
        /// <param name="fileName">Имя файла для сохранения.</param>
        public void SaveExpToFile(String fileName)
        {
            var fi = new FileInfo(fileName);
            var fs = fi.Open(FileMode.Create, FileAccess.Write);
            var bw = new BinaryWriter(fs);
            bw.Write(_net.LayersCount); //записываем число слоев
            for (int i=0; i<_net.LayersCount; ++i)
                bw.Write(_net[i].NeuronsCount); //записываем число нейронов для каждого слоя
            for (int i=0; i<_net.LayersCount; ++i)
            {
                for (int j=0; j<_net[i].NeuronsCount; ++j)
                {
                    bw.Write(_net[i][j].Offset); //записываем смещение
                    foreach (double t in _net[i][j].Weights)
                        bw.Write(t); //записываем весовые коэффициенты
                }
            }
            fs.Flush();
            bw.Close();
        }
#endregion
    }
}
