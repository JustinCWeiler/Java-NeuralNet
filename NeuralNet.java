import java.util.ArrayList;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.json.simple.*;
import org.json.simple.parser.*;

public class NeuralNet
{
	private Matrix[] weights;
	private Vector[] biases;

	private Matrix.Function act;
	private Matrix.Function d_act;

	private double learningRate;
	private boolean type;

	private void NeuralNet(int[] neuronsPerLayer, boolean type, double lr)
	{
		weights = new Matrix[neuronsPerLayer.length - 1];
		biases = new Vector[neuronsPerLayer.length - 1];
		for(int L = 0; L < neuronsPerLayer.length-1; L++)
		{
			weights[L] = new Matrix(neuronsPerLayer[L+1], neuronsPerLayer[L]);
			biases[L] = new Vector(neuronsPerLayer[L+1]);
			weights[L].randomize(-1, 1);
			biases[L].randomize(-1, 1);
		}

		if(type) // True if classification, false if regression
		{
			act = NeuralNet::sig;
			d_act = NeuralNet::d_sig;
		}
		else
		{
			act = x -> x;
			d_act = x -> 1;
		}
		this.type = type;

		learningRate = lr;
	}

	public NeuralNet(int[] neuronsPerLayer, boolean type, double lr)
	{
		NeuralNet(neuronsPerLayer, type, lr);
	}

	public NeuralNet(int[] neuronsPerLayer, boolean type)
	{
		this(neuronsPerLayer, type, 0.1);
	}

	public NeuralNet(String fileName)
	{
		try
		{
			String text = new String(Files.readAllBytes(Paths.get(fileName)));

			try
			{
				JSONObject data = (JSONObject)(new JSONParser()).parse(text);

				// Type
				type = (boolean)data.get("type");
				if(type) // True if classification, false if regression
				{
					act = NeuralNet::sig;
					d_act = NeuralNet::d_sig;
				}
				else
				{
					act = x -> x;
					d_act = x -> 1;
				}

				// Learning Rate
				learningRate = (double)data.get("learning_rate");

				// Weights
				ArrayList<ArrayList<ArrayList<Double>>> ws = (ArrayList<ArrayList<ArrayList<Double>>>)data.get("weights");
				weights = new Matrix[ws.size()];
				for(int L = 0; L < weights.length; L++)
				{
					Matrix w = new Matrix(ws.get(L).size(), ws.get(L).get(0).size());
					for(int j = 0; j < w.getRows(); j++)
					{
						for(int k = 0; k < w.getCols(); k++)
						{
							w.set(j, k, ws.get(L).get(j).get(k));
						}
					}
					weights[L] = w;
				}

				// Biases
				ArrayList<ArrayList<Double>> bs = (ArrayList<ArrayList<Double>>)data.get("biases");
				biases = new Vector[bs.size()];
				for(int L = 0; L < biases.length; L++)
				{
					Vector b = new Vector(bs.get(L).size());
					for(int j = 0; j < b.getRows(); j++)
					{
						b.set(j, bs.get(L).get(j));
					}
					biases[L] = b;
				}
			}
			catch(ParseException e)
			{
				System.out.println("Error: file format incorrect. Reverting to basic setup of a 2-1 classification");
				NeuralNet(new int[] { 2, 1 }, true, 0.1);
			}
		}
		catch(IOException e)
		{
			System.out.println("Error: file not found. Reverting to basic setup of a 2-1 classification");
			NeuralNet(new int[] { 2, 1 }, true, 0.1);
		}
	}

	public void train(Vector[] inputs, Vector[] outputs)
	{
		Matrix[] changePerWeight = new Matrix[weights.length];
		Vector[] changePerBias = new Vector[biases.length];
		// Loop through each input and output and adjust the weights and biases based on those
		for(int n = 0; n < inputs.length; n++)
		{
			Vector[] a = guessTrain(inputs[n]);

			// Initialize dCdA to derivative of error squared
			Vector dCdA = new Vector(a[a.length-1]);
			dCdA.subtract(outputs[n]);
			dCdA.scalarMultiply(2);
			// Loop through the neuron layers backwards
			for(int L = weights.length; L > 0; L--)
			{
				// Compute useful values
				Vector dAdZ = a[L];
				dAdZ.map(d_act);

				Vector[] dZdW = new Vector[weights[L-1].getCols()];
				for(int k = 0; k < dZdW.length; k++)
				{
					dZdW[k] = new Vector(weights[L-1].getRows(), a[L-1].get(k));
				}

				Vector dZdB = new Vector(biases[L-1].getRows(), 1);

				// Compute actual cost differences
				Matrix dCdW = new Matrix(weights[L-1].getRows(), weights[L-1].getCols());
				for(int k = 0; k < dCdW.getCols(); k++)
				{
					dCdW.setCol(k, Vector.hadamardProduct(new Vector[] { dZdW[k], dAdZ, dCdA }));
				}

				Vector dCdB = Vector.hadamardProduct(new Vector[] { dZdB, dAdZ, dCdA });

				dCdW.scalarMultiply(learningRate);
				dCdB.scalarMultiply(learningRate);

				// Add the values to the array to be averaged later
				if(n == 0)
				{
					changePerWeight[L-1] = new Matrix(dCdW);
					changePerBias[L-1] = new Vector(dCdB);
				}
				else
				{
					changePerWeight[L-1].add(dCdW);
					changePerBias[L-1].add(dCdB);
				}

				// Calculate dCdA for the next layer
				Vector[] dZdA = new Vector[a[L-1].getRows()];
				for(int k = 0; k < dZdA.length; k++)
				{
					dZdA[k] = weights[L-1].getCol(k);
				}

				double[] prev_dCdA = new double[a[L-1].getRows()];
				for(int k = 0; k < prev_dCdA.length; k++)
				{
					prev_dCdA[k] = Vector.dot(new Vector(1, a[L-1].getRows()), Vector.hadamardProduct(new Vector[] { dZdA[k], dAdZ, dCdA} ));
				}

				dCdA = new Vector(prev_dCdA);
			}
		}

		for(int L = 0; L < changePerWeight.length; L++)
		{
			changePerWeight[L].scalarDivide(inputs.length);
			changePerBias[L].scalarDivide(inputs.length);
			weights[L].subtract(changePerWeight[L]);
			biases[L].subtract(changePerBias[L]);
		}
	}

	public Vector[] guessTrain(Vector inp)
	{
		Vector[] out = new Vector[weights.length + 1];
		out[0] = inp;
		for(int L = 0; L < weights.length; L++)
		{
			out[L+1] = Vector.multiply(weights[L], out[L]);
			out[L+1].add(biases[L]);
			out[L+1].map(act);
		}
		return out;
	}

	public Vector guess(Vector inp)
	{
		for(int L = 0; L < weights.length; L++)
		{
			inp = Vector.multiply(weights[L], inp);
			inp.add(biases[L]);
			inp.map(act);
		}
		return inp;
	}

	public double meanSquaredError(Vector[] inputs, Vector[] outputs)
	{
		double sum = 0;
		for(int n = 0; n < inputs.length; n++)
		{
			Vector v = guess(inputs[n]);
			v.subtract(outputs[n]);
			sum += v.magnitude()*v.magnitude();
		}
		return sum/inputs.length;
	}
	
	public double rootMeanSquaredError(Vector[] inputs, Vector[] outputs)
	{
		return Math.sqrt(meanSquaredError(inputs, outputs));
	}

	public String getFileName()
	{
		StringBuilder fn = new StringBuilder();

		if(type)
			fn.append("class");
		else
			fn.append("reg");

		fn.append("_");

		fn.append(weights[0].getCols());
		for(int i = 0; i < biases.length; i++)
		{
			fn.append("-");
			fn.append(biases[i].getRows());
		}
		fn.append(".json");

		return fn.toString();
	}

	public void saveNetwork(String fileName)
	{
		//  U+1234
		if(!fileName.equals("ሴ"))
			fileName = fileName + "_" + getFileName();
		else
			fileName = getFileName();
		
		// DATA START
		JSONObject data = new JSONObject();

		//  Neuron list
		ArrayList<Integer> neurons = new ArrayList<Integer>();
		neurons.add(weights[0].getCols());
		for(int i = 0; i < biases.length; i++)
			neurons.add(biases[i].getRows());
		data.put("neurons", neurons);

		//  Type
		data.put("type", type);

		//  Learning rate
		data.put("learning_rate", learningRate);

		//  Weights
		ArrayList<ArrayList<ArrayList<Double>>> ws = new ArrayList<ArrayList<ArrayList<Double>>>();
		for(int L = 0; L < weights.length; L++)
		{
			Matrix w = weights[L];
			ws.add(new ArrayList<ArrayList<Double>>());
			for(int j = 0; j < w.getRows(); j++)
			{
				ws.get(L).add(new ArrayList<Double>());
				for(int k = 0; k < w.getCols(); k++)
				{
					ws.get(L).get(j).add(w.get(j, k));
				}
			}
		}
		data.put("weights", ws);

		//  Biases
		ArrayList<ArrayList<Double>> bs = new ArrayList<ArrayList<Double>>();
		for(int L = 0; L < weights.length; L++)
		{
			Vector b = biases[L];
			bs.add(new ArrayList<Double>());
			for(int j = 0; j < b.getRows(); j++)
			{
				bs.get(L).add(b.get(j));
			}
		}
		data.put("biases", bs);
		// DATA END
		
		// Actually writing the thing
		try
		{
			FileWriter fw = new FileWriter(fileName);
			fw.write(data.toString());
			fw.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public void saveNetwork()
	{
		// U+1234
		saveNetwork("ሴ");
	}

	public Matrix[] getWeights() { return weights; }
	public Vector[] getBiases() { return biases; }

	public static double sig(double x) { return 1/(1+Math.exp(-x)); }
	public static double d_sig(double x) { return x*(1-x); }

	public static double relu(double x) { return x>0 ? x : 0; }
	public static double d_relu(double x) { return x>0 ? 1 : 0; }
}
