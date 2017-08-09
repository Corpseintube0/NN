using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neuronets
{
    /// <summary>
    /// Представляет входной слой нейронов.
    /// </summary>
    public class InputLayer : Layer
    {
        /// <summary>
        /// Инициализирует входной слой нейронов.
        /// </summary>
        /// <param name="links">Связи с последующим слоем.</param>
        /// <param name="layerFunc">Активационная передаточная функция слоя.</param>
        public InputLayer(bool[,] links, FunctionType layerFunc) : base(links, layerFunc)
        {
            for (int i=0; i<NeuronsCount; ++i)
            {
                this[i].SetInputsNumber(1);
                for (int j = 0; j < this[i].Weights.Length; ++j)
                    this[i].Weights[j] = 1.0;
                this[i].Offset = 0.0;
            }
        }

        /// <summary>
        /// Инициализирует входной слой нейронов.
        /// </summary>
        /// <param name="neuronCount">Число нейронов.</param>
        /// <param name="func">Активационная передаточная функция слоя.</param>
        public InputLayer(int neuronCount, FunctionType func) : base(neuronCount, func)
        {
            for (int i = 0; i < NeuronsCount; ++i)
            {
                this[i].SetInputsNumber(1);
                for (int j = 0; j < this[i].Weights.Length; ++j)
                    this[i].Weights[j] = 1.0;
                this[i].Offset = 0.0;
            }
        }
    }
}
