package MNISTReading;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Stores Mnist data with a float 2D array of grayscale values representing the image
 * and its corresponding label (the desired output) as an int.
 */
public class MnistMatrix {

    private float [][] data;
    private int nRows;
    private int nCols;
    private int label;

    /**
     * Constructor call initializing the 2D array that will hold the grayscale values.
     * Without a label (presumingly to see how a network performs in the 'real world')
     * @param nRows: how many pixel rows to initialize.
     * @param nCols: how many pixel columns to initialize.
     */
    MnistMatrix(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;

        this.data = new float[nRows][nCols];
    }
    
    /**
     * Constructor call initializing the 2D array that will hold the grayscale values.
     * as well as storing the label.
     * @param nRows: how many pixel rows to initialize.
     * @param nCols: how many pixel columns to initialize.
     * @param label: the desired output for that image.
     */
    MnistMatrix(int nRows, int nCols, int label) {
        this.nRows = nRows;
        this.nCols = nCols;
        this.label = label;
        
        this.data = new float[nRows][nCols];
    }
    
    /**
     * Returns the full image as a INDArray with r rows and c columns.
     * @return INDArray of grayscale values.
     */
    public INDArray getINDMatrix() {
    	INDArray image_matrix = Nd4j.createFromArray(this.data);
    	return image_matrix;
    }
    
    /**
     *  Stacks the data into one column that is c*r long so that it can be read by the network.
     *  returning a INDArray that had c*r rows and 1 column.
     */
    public INDArray getINDArray() {
    	float[] vector = new float[this.nRows*this.nCols];
    	int index = 0;
    	
    	for (int i = 0; i < this.data[0].length; i++) { //columns
    		for (int j = 0; j < this.data.length; j++) { //rows
    			vector[index] = this.data[i][j];
    			index++;
    		}
    		
    	}
    	
    	return Nd4j.create(vector, new int[] {this.nRows*this.nCols,1});
    }
    
    public INDArray getExpectedOut() {
    	INDArray expected = Nd4j.zeros(10,1);
    	expected.put(this.label, 1, (float) 1.0);
    	
    	return expected;
    }

    public void setValue(int row, int col, int value) {
    	this.data[row][col] = value;
    }

    public float getValue(int r, int c) {
    	return this.data[r][c];
    }
    
	public void setLabel(int label) {
		this.label = label;
	}
	
    public int getLabel() {
        return this.label;
    }
    
    public int getNumberOfRows() {
        return this.nRows;
    }
   
    public int getNumberOfColumns() {
        return this.nCols;
    }

}
