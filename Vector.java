public class Vector extends Matrix
{
	public Vector(double[] v)
	{
		super(v);
	}

	public Vector(int m)
	{
		super(m, 1);
	}

	public Vector(int m, double s)
	{
		this(m);
		for(int j = 0; j < m; j++)
		{
			set(j, s);
		}
	}

	public Vector(Vector v)
	{
		this(v.getRows());
		for(int j = 0; j < v.getRows(); j++)
		{
			set(j, v.get(j));
		}
	}

	public static Vector multiply(Matrix m, Vector v)
	{
		Matrix n = Matrix.multiply(m, v);

		double[] d = new double[n.getRows()];
		for(int i = 0; i < d.length; i++)
		{
			d[i] = n.get(i, 0);
		}

		return new Vector(d);
	}

	public double get(int m)
	{
		return super.get(m, 0);
	}

	public void set(int m, double d)
	{
		super.set(m, 0, d);
	}

	public double magnitude()
	{
		double sum = 0;
		for(int k = 0; k < getCols(); k++)
		{
			sum += get(k)*get(k);
		}
		return Math.sqrt(sum);
	}

	public static Vector hadamardProduct(Vector v1, Vector v2)
	{
		Matrix n = Matrix.hadamardProduct(v1, v2);

		double[] d = new double[n.getRows()];
		for(int i = 0; i < d.length; i++)
		{
			d[i] = n.get(i, 0);
		}

		return new Vector(d);
	}

	public static Vector hadamardProduct(Vector[] vectors)
	{
		Vector end = new Vector(vectors[0]);
		for(int n = 1; n < vectors.length; n++)
		{
			end = hadamardProduct(end, vectors[n]);
		}

		return end;
	}

	public static double dot(Vector v1, Vector v2)
	{
		double sum = 0;
		for(int i = 0; i < v1.getRows(); i++)
		{
			sum += v1.get(i) * v2.get(i);
		}
		return sum;
	}
}
