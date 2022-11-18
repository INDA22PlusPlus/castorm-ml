// See https://aka.ms/new-console-template for more information
using MNIST;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
Console.WriteLine("Hello, World!");



var nn = new NeuralNetwork();
nn.AddLayer(new InputLayer(1)).AddLayer(new FullyConnectedLayer(1, 1))
    .AddLayer(new OutputLayer(1));

//Console.WriteLine(nn.Forwardprop(Vector<double>.Build.Dense(1, 0.88)));
//nn.Print();
//var nnn = nn.Mutate();
//Console.WriteLine(nnn.Forwardprop(Vector<double>.Build.Dense(1, 0.88)));
//nnn.Print();

//var expected = Vector<double>.Build.Dense(1, 0);
//Console.WriteLine("Cost NN: " + nn.Cost(expected));
//Console.WriteLine("Cost NNN: " + nnn.Cost(expected));



var inp = new Vector<double>[1000];
var outp = new Vector<double>[1000];

for (int i = 0; i < 1000; i++)
{
    var val = new Random().NextDouble();
    var v = Vector<double>.Build.Dense(1, val);
    inp[i] = v;
    outp[i] = Vector<double>.Build.Dense(1, 0);
}

var test = new Vector<double>[] { Vector<double>.Build.Dense(1, 0.5) };
var expected = new Vector<double>[] { Vector<double>.Build.Dense(1, 0) };
Console.WriteLine(nn.Cost(test, expected));
var res = nn.Train(inp, outp, 10000);
Console.WriteLine(res.Cost(test, expected));

Console.WriteLine(res.Forwardprop(Vector<double>.Build.Dense(1, 0.88)));
Console.WriteLine(res.Forwardprop(Vector<double>.Build.Dense(1, 0.03)));
Console.WriteLine(res.Forwardprop(Vector<double>.Build.Dense(1, 0.44)));
Console.WriteLine(res.Forwardprop(Vector<double>.Build.Dense(1, 0.56)));
