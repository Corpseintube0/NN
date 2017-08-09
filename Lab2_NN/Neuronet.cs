using System;

namespace Neuronets
{
    //TODO: 1)Оставшиеся активационные функции 2) сохранение структуры сети в файл (без коэффициентов)
    /// <summary>
    /// Искуственная нейронная сеть типа "многослойный персептрон".
    /// </summary>
    public class Neuronet
    {
        #region Поля

        /// <summary>
        /// Массив с слоями нейронной сети.
        /// </summary>
        private Layer[] _layers;

        #endregion

        #region Конструкторы

        /// <summary>
        /// Инициализирует экземпляр класса Neuronet без связей.
        /// </summary>
        /// <param name="layersInitiator">Размерность массива - число слоёв, значения элементов массива - число нейронов.</param>
        /// <param name="defaultFunc">Активационная функция по-умолчанию.</param>
        public Neuronet(int[] layersInitiator, FunctionType defaultFunc = FunctionType.Linear)
        {
            _layers = new Layer[layersInitiator.Length];
            _layers[0] = new InputLayer(layersInitiator[0], defaultFunc);
            for (int i = 1; i < _layers.Length; ++i)
                _layers[i] = new Layer(layersInitiator[i], defaultFunc);
        }

        #endregion

        #region Методы

        /// <summary>
        /// Формирует связи указанного слоя со следующим слоем.
        /// </summary>
        /// <param name="layerIndex">Индекс слоя.</param>
        /// <param name="links">Матрица связей.</param>
        public void SetLinks(int layerIndex, bool[,] links)
        {
            if (_layers[layerIndex].NeuronsCount != links.GetLength(0))
                throw new Exception("Размер матрицы:" + links.GetLength(0) + "x" + links.GetLength(1) + ", число нейронов: " + _layers[layerIndex].NeuronsCount);
            if (layerIndex != _layers.Length - 1)
                if (_layers[layerIndex+1].NeuronsCount != links.GetLength(1))
                    throw new Exception("Размер матрицы:" + links.GetLength(0) + "x" + links.GetLength(1) + ", число нейронов: " + _layers[layerIndex+1].NeuronsCount);
            
            if (layerIndex == 0)
                _layers[layerIndex] = new InputLayer(links, _layers[layerIndex].LayerFunction);
            else
                _layers[layerIndex] = new Layer(links, _layers[layerIndex].LayerFunction);

            //входной(первый) слой
            if (layerIndex == 0) 
            {
                for (int i = 0; i < _layers[layerIndex].NeuronsCount; ++i)
                {
                    for (int j = 0; j < _layers[layerIndex][i].Weights.Length; ++j)
                        _layers[layerIndex][i].Weights[j] = 1.0;
                }
                return;
            }
            //выходной(последний) и скрытые слои
            for (int i = 0; i < _layers[layerIndex].NeuronsCount; ++i)
            {
                //выставляем число входов для нейронов
                int inputCount = 0;
                for (int j=0; j < _layers[layerIndex-1].Links.GetLength(1); ++j)
                {
                    inputCount = 0;
                    for (int k=0; k < _layers[layerIndex-1].Links.GetLength(0); ++k)
                    {
                        if (_layers[layerIndex - 1].Links[k, j])
                            ++inputCount;
                    }
                }
                _layers[layerIndex][i].SetInputsNumber(inputCount);
            }
            if (layerIndex != LayersCount - 1)
                _layers[layerIndex + 1] = new Layer(_layers[layerIndex], _layers[layerIndex + 1].Links,
                                                    _layers[layerIndex + 1].LayerFunction);
        }

        /// <summary>
        /// Связывает слои сети ВСЕМИ возможными связями.
        /// </summary>
        public void SetLinks()
        {
            for (int i=0; i < LayersCount - 1; ++i)
            {
                var newLinks = new bool[_layers[i].NeuronsCount, _layers[i+1].NeuronsCount];
                for (int j = 0; j < _layers[i].NeuronsCount; ++j )
                    for (int k = 0; k < _layers[i + 1].NeuronsCount; ++k)
                        newLinks[j, k] = true;
                _layers[i].Links = (bool[,])newLinks.Clone();
            }
            
            //для последнего слоя
            OutputLayer.Links = new bool[OutputLayer.NeuronsCount,1];
            for (int i=0; i<OutputLayer.NeuronsCount; ++i)
                OutputLayer.Links[i,0] = true;

            //выставляем InputCount
            for (int i=1; i < LayersCount; ++i)
                for(int j = 0; j<_layers[i].NeuronsCount; ++j)
                    _layers[i][j].SetInputsNumber(_layers[i - 1].NeuronsCount);
        }

        /// <summary>
        /// Активирует сеть входным импульсом.
        /// </summary>
        /// <param name="input">Входной импульс сети.</param>
        /// <returns>Результат работы сети.</returns>
        public double[] Activate(double[] input)
        {
            var newInput = new double[1, input.Length];
            for (int i = 0; i < newInput.Length; ++i)
                newInput[0, i] = input[i];
            _layers[0].RecieveSignals(newInput); //передаем входной сигнал на первый слой

            if (_layers.Length < 2)
                return _layers[_layers.Length - 1].FlashSignals();

            for (int ind = 1; ind < _layers.Length; ++ind )
            {
                var temp = _layers[ind - 1].TransferSignals();
                _layers[ind].RecieveSignals(temp);
            }

            return _layers[_layers.Length - 1].FlashSignals();
        }

        #endregion

        #region Индексаторы

        /// <summary>
        /// Индексатор для нейросети.
        /// </summary>
        /// <param name="index">Индекс слоя.</param>
        /// <returns>Слой сети с указанным индексом.</returns>
        public Layer this[int index]
        {
            get
            {
                return _layers[index];
            }
        }

        #endregion

        #region Свойства

        /// <summary>
        /// Получает ссылку на массив слоёв.
        /// </summary>
        public Layer[] Layers
        {
            get
            {
                return _layers;
            }
        }

        /// <summary>
        /// Получает число слоев данной нейросети.
        /// </summary>
        public int LayersCount
        {
            get
            {
                return _layers.Length;
            }
        }

        /// <summary>
        /// Получает ссылку на последний(выходной) слой нейронов.
        /// </summary>
        public Layer OutputLayer
        {
            get
            {
                return _layers[_layers.Length - 1];
            }
        }

        #endregion
    }
}
