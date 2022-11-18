using MathNet.Numerics.LinearAlgebra;

namespace MNIST;

public static class ActivationFunctionExtentions
{
    public static Func<double, double> GetActivationFunction(this ActivationFunction f) => f switch
    {
        ActivationFunction.Sigmoid => input => 1.0 / (1.0 + Math.Exp(-input)),
        ActivationFunction.ReLU => input => Math.Max(0, input),
        ActivationFunction.Linear => input => input,
        _ => throw new NotImplementedException()
    };

    public static Func<double, double> GetBackpropFunction(this ActivationFunction f) => f switch
    {
        ActivationFunction.Sigmoid => input => Math.Exp(-input) / Math.Pow(1.0 + Math.Exp(-input), 2),
        ActivationFunction.Linear => input => 1.0,
        ActivationFunction.ReLU => input => input > 0 ? 1.0 : 0.0,
        _ => throw new NotImplementedException()
    };
}
public enum ActivationFunction
{
    Sigmoid,
    ReLU,
    Linear
}

public enum LayerType
{
    Input,
    FullyConnected,
    Output
}

public class NeuralNetwork
{
    public static double LearningRate = 0.1;
    public List<Layer> Layers = new();
    public Vector<double>? Output;
    public NeuralNetwork AddLayer(Layer layer)
    {
        Layers.Add(layer);
        return this;
    }

    public Vector<double> ExecuteTest()
    {
        var v = Vector<double>.Build.Random(10);
        Console.WriteLine(v);
        return Forwardprop(v);
    }

    public Vector<double> Forwardprop(Vector<double> input)
    {
        if (Layers.Count == 0) return input.Clone();
        Layer curr = Layers[0];
        Vector<double> output = curr.ForwardProp(input);
        output.Map(curr.ActivationFunction.GetActivationFunction());
        for (int i = 1; i < Layers.Count; i++)
        {
            curr = Layers[i];
            output = curr.ForwardProp(output);
            output.Map(curr.ActivationFunction.GetActivationFunction());
        }
        Output = output;
        return output;
    }

    public double Cost(Vector<double> expected)
    {
        return Output?.Subtract(expected).Map(d => d * d).Sum() ?? throw new Exception("Forwardprop never called!");
    }

    public double Cost(Vector<double>[] input, Vector<double>[] expected)
    {
        double res = 0;
        for (int i = 0; i < input.Length; i++)
        {
            Forwardprop(input[i]);
            res += Cost(expected[i]);
        }
        return res / input.Length;
    }

    public void Backprop(Vector<double> expected)
    {

    }

    public NeuralNetwork Mutate()
    {
        var nn = new NeuralNetwork();
        foreach (var l in Layers)
        {
            nn.AddLayer(l.Mutate());
        }
        return nn;
    }

    public NeuralNetwork Train(Vector<double>[] input, Vector<double>[] expected, int iter = 1000)
    {
        double best = Cost(input, expected);
        NeuralNetwork nn = this;
        for (int i = 0; i < iter; i++)
        {
            LearningRate = 0.2 / i;
            var nnn = Mutate();
            var cost = nnn.Cost(input, expected);
            if (cost < best)
            {
                best = cost;
                nn = nnn;
            }
        }
        LearningRate = 0.2;
        return nn;
    }

    public void Print()
    {
        Console.WriteLine("NN: ");
        foreach (var l in Layers)
        {
            Console.WriteLine(l.Neurons);
        }
        Console.Write("-----");
    }
}

public abstract class Layer
{
    public LayerType LayerType;
    public ActivationFunction ActivationFunction;
    public Vector<double> Neurons;
    public Layer(int size, LayerType layerType, ActivationFunction activationFunction)
    {
        LayerType = layerType;
        ActivationFunction = activationFunction;
        Neurons = Vector<double>.Build.Dense(size);
    }

    public abstract Vector<double> ForwardProp(Vector<double> input);
    public abstract Vector<double> Backprop(Vector<double> input);
    public abstract Layer Mutate();
}

public class FullyConnectedLayer : Layer
{
    public Matrix<double> Weights;
    public Vector<double> Biases;
    public FullyConnectedLayer(int inputSize, int outputSize) : base(inputSize, LayerType.FullyConnected, ActivationFunction.Sigmoid)
    {
        Weights = Matrix<double>.Build.Random(outputSize, inputSize);
        Biases = Vector<double>.Build.Random(outputSize);
    }

    public override Vector<double> Backprop(Vector<double> input)
    {
        throw new NotImplementedException();
    }

    public override Vector<double> ForwardProp(Vector<double> input)
    {
        var v = Weights.Multiply(input).Add(Biases);
        Neurons = v.Clone();
        return v;
    }

    public override Layer Mutate()
    {
        var l = new FullyConnectedLayer(Weights.ColumnCount, Weights.RowCount);
        l.Weights = Weights.Clone();
        l.Biases = Biases.Clone();
        l.Weights = l.Weights.Map(v => v + (2 * new Random().NextDouble() - 1) * NeuralNetwork.LearningRate);
        l.Biases = l.Biases.Map(v => v + (2 * new Random().NextDouble() - 1) * NeuralNetwork.LearningRate);
        return l;
    }
}

public class InputLayer : Layer
{
    public InputLayer(int size) : base(size, LayerType.Input, ActivationFunction.Linear)
    {

    }

    public override Vector<double> Backprop(Vector<double> input)
    {
        throw new NotImplementedException();
    }

    public override Vector<double> ForwardProp(Vector<double> input)
    {
        var c = input.Clone();
        return c.Map(ActivationFunction.GetActivationFunction());
    }

    public override Layer Mutate()
    {
        return new InputLayer(Neurons.Count)
        {
            Neurons = Neurons.Clone()
        };
    }
}

public class OutputLayer : Layer
{
    public OutputLayer(int size) : base(size, LayerType.Output, ActivationFunction.Sigmoid)
    {

    }

    public override Vector<double> Backprop(Vector<double> input)
    {
        throw new NotImplementedException();
    }

    public override Vector<double> ForwardProp(Vector<double> input)
    {
        var clone = input.Clone();
        return clone.Map(ActivationFunction.GetActivationFunction());
    }

    public override Layer Mutate()
    {
        return new OutputLayer(Neurons.Count)
        {
            Neurons = Neurons.Clone()
        };
    }
}

public class Neuron
{
    public double Value { get; set; }
}

public class Weight
{
    public double Value { get; set; }
    public static Weight Random => new()
    {
        Value = new Random().NextDouble()
    };

    public static Weight Zero => new()
    {
        Value = 0
    };
}

public class Bias
{
    public double Value { get; set; }
    public static Bias Random => new()
    {
        Value = new Random().NextDouble()
    };
}

