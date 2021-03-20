public class NeuralTest
{
	// z = x^2 + y^2
	
	private static final int XMAX = 10;
	private static final int YMAX = 10;
	
	/*
	private static Vector[] in = new Vector[XMAX*YMAX];
	private static Vector[] ou = new Vector[XMAX*YMAX];
	*/

	private static Vector[] in = {new Vector(new double[]{1, 4})};
	private static Vector[] ou = {new Vector(new double[]{17})};
	
	public static void setup()
	{
		/*
		for(int i = 0; i < XMAX; i++)
		{
			for(int j = 0; j < YMAX; j++)
			{
				in[YMAX*i+j] = new Vector(new double[]{i, j});
				ou[YMAX*i+j] = new Vector(new double[]{i*i + j*j});
			}
		}
		*/
	}

	public static void main(String[] args)
	{
		NeuralNet net = new NeuralNet(new int[]{2, 1}, false, 0.01);
		setup();

		double guess = 0;
		int i = 0;
		while(guess <= 17-0.1 || guess >= 17+0.1)
		{
			net.train(in, ou);
			guess = net.guess(new Vector(new double[]{1, 4})).get(0);
			System.out.println(guess);
			i++;
		}

		System.out.println(i);
	}
}

