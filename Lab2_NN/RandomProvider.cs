using System;
using System.Threading;

public static class RandomProvider
{
    private static int _seed = Environment.TickCount;

    private static ThreadLocal<Random> randomWrapper = new ThreadLocal<Random>(() => new Random(Interlocked.Increment(ref _seed)));

    public static Random GetThreadRandom()
    {
        return randomWrapper.Value;
    }
}