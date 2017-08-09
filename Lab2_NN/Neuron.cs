using System;

namespace Neuronets
{
    /// <summary>
    /// Модель формального нейрона.
    /// </summary>
    public class Neuron
    {
        #region Поля

        /// <summary>
        /// Коэффициенты весов входных сигналов данного нейрона.
        /// </summary>
        public double[] Weights;

        /// <summary>
        /// Экземпляр делегата ActivationFuncDelegate для вызова нужной активационной функции.
        /// </summary>
        private ActivationFuncDelegate _afd;

        /// <summary>
        /// Параметр активационной функции нейрона.
        /// </summary>
        private double _param;

        /// <summary>
        /// Входные немодифицированные сигналы.
        /// </summary>
        public double[] SignalsIn;

        /// <summary>
        /// Тип активационной функции данного нейрона.
        /// </summary>
        private FunctionType _activationFunc;

        #endregion

        #region Конструкторы

        /// <summary>
        /// Инициализирует экземпляр класса Neuron с нулевыми значениями входных сигналов и случайными начальными весами.
        /// </summary>
        /// <param name="signalsCount">Число синапсов на входе нейрона.</param>
        /// <param name="ft">Тип активационной функции нейрона.</param>
        /// <param name="funcParam">Значение параметра активационной функции.</param>
        public Neuron(int signalsCount, FunctionType ft, double funcParam)
        {
            SignalsIn = new double[signalsCount];
            Weights = new double[signalsCount];
            _param = funcParam;
            var r = RandomProvider.GetThreadRandom();

            Offset = Convert.ToDouble(r.Next(-100, 100)) / 1000;
            for (int i = 0; i < SignalsIn.Length; ++i)
            {
                Weights[i] = Convert.ToDouble(r.Next(-100, 100)) / 1000; //инициализация весов
                SignalsIn[i] = 0;
            }
            _activationFunc = ft;
            switch (ft) //инициализация активационной функции нейрона
            {
                case FunctionType.Linear:
                    _afd = LinearFunc; //линейная
                break;
                case FunctionType.Sigmoid:
                    _afd = LogSigmoidFunc; //лог-сигмоидальная
                    break;
                case FunctionType.HyperbolicTangent:
                    _afd = HyperbolicTangentFunc; //гиперболический тангенс
                    break;
                case FunctionType.Threshold:
                    _afd = ThresholdFunc; //пороговая
                    break;
                default:
                    throw new Exception("Неизвестный тип активационной функции");
            }
        }

        #endregion

        #region Активационные функции

        /// <summary>
        /// Лог-сигмоидная активационная функция нейрона.
        /// </summary>
        /// <param name="a">Значение параметра функции.</param>
        /// <returns>Значение активационной функции в диапазоне [0; 1].</returns>
        private double LogSigmoidFunc(double a)
        {
            return 1.0/(1 + Math.Exp(-1*a*Sum()));
        }

        /// <summary>
        /// Линейная функция вида f(S) = aS.
        /// </summary>
        /// <param name="a">Значение линейного коэффициента.</param>
        /// <returns>Значение активационной функции в диапазоне [-∞; +∞].</returns>
        private double LinearFunc(double a)
        {
            return Sum() * a;
        }

        /// <summary>
        /// Пороговая активационная функция нейрона.
        /// </summary>
        /// <param name="d">Значение порога функции.</param>
        /// <returns>0 или 1.</returns>
        private double ThresholdFunc(double d)
        {
            return Sum() > d ? 1 : 0;
        }

        /// <summary>
        /// Гиперболический тангенс, активационная функция нейрона.
        /// </summary>
        /// <param name="a">Значение параметра функции.</param>
        /// <returns>Значение активационной функции в диапазоне [-1; 1].</returns>
        private double HyperbolicTangentFunc(double a)
        {
            double s = Sum();
            return (Math.Exp(a*s) - Math.Exp(-1*a*s)) / (Math.Exp(a*s) + Math.Exp(-1*a*s));
        }

        #endregion

        #region  Делегаты

        /// <summary>
        /// Делегат для инкапсуляции активационной функции.
        /// </summary>
        private delegate double ActivationFuncDelegate(double arg);

        #endregion

        #region Свойства

        /// <summary>
        /// Получает или задает значение параметра активационной функции нейрона.
        /// </summary>
        public double ParamValue
        {
            get
            {
                return _param;
            }
            set
            {
                _param = value;
            }
        }

        /// <summary>
        /// Получает или задает коэффициент смещения нейрона.
        /// </summary>
        public double Offset
        {
            get;
            set;
        }

        /// <summary>
        /// Возвращает или задаёт тип активационной функции для данного нейрона.
        /// </summary>
        public FunctionType ActivationFunc
        {
            get
            {
                return _activationFunc;
            }
            set
            {
                _activationFunc = value;
                switch (value)
                {
                    case FunctionType.Linear:
                        _afd = LinearFunc; //линейная
                        break;
                    case FunctionType.Sigmoid:
                        _afd = LogSigmoidFunc; //лог-сигмоидальная
                        break;
                    case FunctionType.HyperbolicTangent:
                        _afd = HyperbolicTangentFunc; //гиперболический тангенс
                        break;
                    case FunctionType.Threshold:
                        _afd = ThresholdFunc; //пороговая
                        break;
                    default:
                        throw new Exception("Неизвестный тип активационной функции");
                }
            }
        }

        #endregion

        #region Методы

        /// <summary>
        /// Сброс весов нейрона и присвоение им новых случайных значений.
        /// </summary>
        private void ResetWeights()
        {
            var r = RandomProvider.GetThreadRandom();
            for (int i=0; i<Weights.Length; ++i)
                Weights[i] = Convert.ToDouble(r.Next(-100, 100)) / 1000;
        }

        /// <summary>
        /// Сумматор входных сигналов нейрона. Нужен для передачи значения в активационную функцию.
        /// </summary>
        /// <returns>Сумма значений входных сигналов, умноженных на весовые коэффициенты.</returns>
        private double Sum()
        {
            double res = 0;
            for (int i = 0; i < SignalsIn.Length; ++i )
                res += SignalsIn[i] * Weights[i];
            return res + Offset;
        }
        
        /// <summary>
        /// Возвращает значение сигнала активационной функции нейрона.
        /// </summary>
        /// <returns>Сигнал, возвращаемый нейроном.</returns>
        public double SignalOut()
        {
            //решение куда дальше передавать сигнал определяется на уровне слоя (класс Layer)
            return _afd(_param);
        }

        /// <summary>
        /// Устанавливает размерность массива входных сигналов для данного нейрона.
        /// </summary>
        /// <param name="arg">Новое количество входов нейрона.</param>
        public void SetInputsNumber(int arg)
        {
            if (arg < 1)
                throw new ArgumentException("Число входных сигналов не может быть меньше (1).", "arg");
            SignalsIn = new double[arg];
            Weights = new double[arg];
            ResetWeights();
        }

        #endregion
    }
}
