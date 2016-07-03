using System;
using System.Collections.Generic;

namespace Neuronets
{
    /// <summary>
    /// Слой нейронов.
    /// </summary>
    public class Layer
    {
        #region Поля

        /// <summary>
        /// Нейроны данного слоя.
        /// </summary>
        private Neuron[] _neurons;

        /// <summary>
        /// Связи со следующим слоем.
        /// </summary>
        private bool[,] _links; //матрица смежности орграфа(теория графов)

        /// <summary>
        /// Активационная функция для слоя.
        /// </summary>
        private FunctionType _activate;


        private double _layerParam;

        #endregion

        #region Свойства

        /// <summary>
        /// Получает или задает активационную функцию для слоя.
        /// </summary>
        public FunctionType LayerFunction
        {
            get
            {
                return _activate;
            }
            set
            {
                foreach (var n in _neurons)
                {
                    _activate = value;
                    n.ActivationFunc = value;
                }
            }
        }

        /// <summary>
        /// Параметр активационной функции по-умолчанию.
        /// </summary>
        public double FuncDefaultParam
        {
            get
            {
                return _layerParam;
            }
            set
            {
                _layerParam = value;
                for (int i = 0; i < NeuronsCount; ++i )
                    _neurons[i].ParamValue = value;
            }
        }

        /// <summary>
        /// Число нейронов, содержащихся в данном слое.
        /// </summary>
        public int NeuronsCount
        {
            get
            {
                return _neurons.Length;
            }
        }

        /// <summary>
        /// Получает массив связей со следующим слоем нейронов.
        /// </summary>
        public bool[,] Links
        {
            get
            {
                return _links;
            }
            internal set
            {
                _links = (bool[,])value.Clone();
            }
        }

        #endregion

        #region Конструкторы

        /// <summary>
        /// Инициализирует экземпляр класса Layer и входы нейронов с помощью связей предыдущего слоя.
        /// </summary>
        /// <param name="prevLayer">Слой-предшественник.</param>
        /// <param name="links">Массив связей со следующим слоем.</param>
        /// <param name="func">Активационная функция для слоя по-умолчанию.</param>
        public Layer(Layer prevLayer, bool[,] links, FunctionType func)
        {
            var oldLinks = prevLayer.Links;
            if (oldLinks.GetLength(1) != links.GetLength(0))
                throw new Exception("Массив связей не соответствует числу нейронов.");
            
            _links = (bool[,])links.Clone();
            _neurons = new Neuron[links.GetLength(0)]; //резервируем место под нейроны
            

            for (int i = 0; i < oldLinks.GetLength(1); ++i) //i - индекс столбца
            {
                int sinapsCount = 0; //счетчик синапсов нейрона
                for (int j = 0; j < oldLinks.GetLength(0); ++j) //j - индекс строки
                {
                    if (oldLinks[j, i])
                        ++sinapsCount;
                }
                _neurons[i] = new Neuron(sinapsCount, func, 1.0);  
            }
            LayerFunction = func;
            FuncDefaultParam = 1.0;
        }

        /// <summary>
        ///  Инициализирует экземпляр класса Layer со связями.
        /// </summary>
        /// <param name="links">Массив связей со следующим слоем.</param>
        /// <param name="func">Активационная функция для слоя по-умолчанию.</param>
        public Layer(bool[,] links, FunctionType func)
        {
            _links = (bool[,])links.Clone();
            _neurons = new Neuron[links.GetLength(0)]; //резервируем место под нейроны
            
            for (int i = 0; i < _neurons.Length; ++i)
                _neurons[i] = new Neuron(1, func, 1.0);
            LayerFunction = func;
            FuncDefaultParam = 1.0;
            
        }

        /// <summary>
        /// Инициализирует экземпляр класса Layer с заданным числом нейронов c одним входным сигналом(используется когда связи неизвестны заранее).
        /// </summary>
        /// <param name="neuronCount">Число нейронов для слоя.</param>
        /// <param name="func">Активационная функция по-умолчанию.</param>
        public Layer(int neuronCount, FunctionType func)
        {
            _neurons = new Neuron[neuronCount]; //сами нейроны в конструкторе слоя не создаются, только резервируется место под них
            
            for (int i = 0; i < _neurons.Length; ++i)
                _neurons[i] = new Neuron(1, func, 1.0);
            LayerFunction = func;
            FuncDefaultParam = 1.0;
        }

        #endregion
        
        #region Индексаторы

        /// <summary>
        /// Индексатор для слоя.
        /// </summary>
        /// <param name="index">Индекс нейрона.</param>
        /// <returns>Нейрон с указанным индексом.</returns>
        public Neuron this[int index]
        {
            get
            {
                return _neurons[index];
            }
        }

        #endregion

        #region Методы

        /// <summary>
        /// Получает сигналы от предыдущего слоя и передает их нейронам данного слоя.
        /// </summary>
        /// <param name="values">Массив значений сигналов.</param>
        public void RecieveSignals(double[,] values)
        {
            //работает в паре с TransferSignals
            if (values.GetLength(1) != _neurons.Length)
                throw new Exception("RecieveSignals: размерность массива не совпадает с числом нейронов.");

            var tempSinapsValues = new List<double>(); //буфер для группировки синапсов одного нейрона
            for (int i = 0; i < values.GetLength(1); ++i) //i - индекс столбца
            {
                 tempSinapsValues.Clear();
                 for (int j = 0; j < values.GetLength(0); ++j) //j - индекс строки
                 {
                     if (!Double.IsNaN(values[j, i])) //если сигнал есть
                         tempSinapsValues.Add(values[j, i]);
                 }
                for (int k = 0; k < tempSinapsValues.Count; ++k)
                    _neurons[i].SignalsIn[k] = tempSinapsValues[k]; //перемещаем значения сигналов из буфера в нейрон
            }
        }

        /// <summary>
        /// Передает сигналы в следующий слой нейронов.
        /// </summary>
        /// <returns>Массив значений сигналов после обработки, распределенный в соответствии со связями нейронов.</returns>
        public double[,] TransferSignals()
        {
            //работает в паре с RecieveSignals
            var res = new double[_neurons.Length, _links.GetLength(1)];
            for (int i = 0; i < res.GetLength(0); ++i)
            {
                for (int j = 0; j < res.GetLength(1); ++j)
                {
                    if (_links[i, j])
                        res[i, j] = _neurons[i].SignalOut();
                    else
                        res[i, j] = Double.NaN; //NaN = отсутствие сигнала
                }
            }
            return res;
        }

        /// <summary>
        /// Возвращает значения сигналов нейронов после обработки.
        /// </summary>
        /// <returns>Одномерный массив значений сигналов нейронов после обработки.</returns>
        public double[] FlashSignals()
        {
            var res = new double[_neurons.Length];
            for (int i = 0; i < _neurons.Length; ++i)
            {
                res[i] = _neurons[i].SignalOut();
            }
            return res;
        }

        #endregion
    }
}
