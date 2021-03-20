// Creates an rows by cols matrix
public class Matrix
{
	private double[][] mat;
	private int rows, cols;

	public interface Function { double run(double x); }

	public Matrix(int m, int n)
	{
		mat = new double[m][n];

		rows = m;
		cols = n;
	}

	// Assume everything is nicely formatted
	public Matrix(double[][] init)
	{
		mat = init;

		rows = init.length;
		cols = init[0].length;
	}

	public Matrix(double[] vector)
	{
		mat = new double[vector.length][1];
		for(int i = 0; i < mat.length; i++)
		{
			set(i, 0, vector[i]);
		}

		rows = vector.length;
		cols = 1;
	}

	public Matrix(Matrix matr)
	{
		mat = matr.getMat();
		rows = matr.getRows();
		cols = matr.getCols();
	}

	public Matrix transpose()
	{
		Matrix end = new Matrix(cols, rows);
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				end.set(j, i, get(i, j));
			}
		}
		return end;
	}

	// Multiply m1 onto m2
	public static Matrix multiply(Matrix m1, Matrix m2)
	{
		double[][] end = new double[m1.getRows()][m2.getCols()];

		for(int i = 0; i < end.length; i++)
		{
			for(int j = 0; j < end[0].length; j++)
			{
				end[i][j] = multiplyPart(m1, m2, i, j);
			}
		}

		return new Matrix(end);
	}
	private static double multiplyPart(Matrix mat1, Matrix mat2, int row, int col)
	{
		double sum = 0;
		for(int i = 0; i < mat1.getCols(); i++)
		{
			double a = mat1.get(row, i);
			double b = mat2.get(i, col);
			sum += a*b;
		}
		return sum;
	}

	public static Matrix hadamardProduct(Matrix m1, Matrix m2)
	{
		Matrix end = new Matrix(m1.getRows(), m1.getCols());
		for(int i = 0; i < end.getRows(); i++)
		{
			for(int j = 0; j < end.getCols(); j++)
			{
				end.set(i, j, m1.get(i, j) * m2.get(i, j));
			}
		}
		return end;
	}

	public void add(Matrix other)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				set(i, j, get(i, j) + other.get(i, j));
			}
		}
	}
	public void subtract(Matrix other)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				set(i, j, get(i, j) - other.get(i, j));
			}
		}
	}

	public void scalarMultiply(double scalar)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				mat[i][j] *= scalar;
			}
		}
	}
	public void scalarDivide(double scalar)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				mat[i][j] /= scalar;
			}
		}
	}
	public void scalarAdd(double scalar)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				mat[i][j] += scalar;
			}
		}
	}
	public void scalarSubtract(double scalar)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				mat[i][j] -= scalar;
			}
		}
	}

	public void set(int m, int n, double d) { mat[m][n] = d; }

	public double[][] getMat() { return mat; }
	public double get(int m, int n) { return mat[m][n]; }
	public int getRows() { return rows; }
	public int getCols() { return cols; }

	public double[] getRow(int m) { return mat[m]; }

	public Vector getCol(int n)
	{
		Vector v = new Vector(mat.length);
		for(int i = 0; i < v.getRows(); i++)
		{
			v.set(i, get(i, n));
		}
		return v;
	}

	public void setCol(int n, Vector v)
	{
		for(int i = 0; i < getRows(); i++)
		{
			set(i, n, v.get(i));
		}
	}

	public void setCol(int n, double[] nums)
	{
		Vector v = new Vector(nums);
		setCol(n, v);
	}

	public void setRow(int m, double[] nums) { mat[m] = nums; }

	public void map(Function f)
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				set(i, j, f.run(get(i, j)));
			}
		}
	}

	public void randomize(double a, double b)
	{

		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				set(i, j, Math.random()*(b - a) + a);
			}
		}
	}

	public void randomize()
	{
		randomize(-1, 1);
	}

	public void print()
	{
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				System.out.print(get(i, j) + "\t");
			}
			System.out.println();
		}
	}
}
