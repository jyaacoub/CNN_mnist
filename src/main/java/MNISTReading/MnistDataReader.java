package MNISTReading;

import java.io.*;
import java.util.ArrayList;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Class to used to read the Mnist data set.
 * @author jyaac
 *
 */
public class MnistDataReader  {
	private String dataFolder = "src/main/resources/";
	
	public String train_filepath = dataFolder + "train-images.idx3-ubyte";
	public String train_filepath_labels = dataFolder + "train-labels.idx1-ubyte";
	public String test_filepath = dataFolder + "t10k-images.idx3-ubyte";
	public String test_filepath_labels = dataFolder + "t10k-labels.idx1-ubyte";
	
	private ArrayList<MnistMatrix> train_set = null;
	private ArrayList<MnistMatrix> test_set = null;
	
	public ArrayList<MnistMatrix> getTrainData() {
		if (this.train_set == null) {
			try { 
		        System.out.println("\nTrainData:");
				this.train_set = readData(train_filepath, train_filepath_labels);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return this.train_set;		
	}

	public ArrayList<MnistMatrix> getTestData() {
		if (this.test_set == null) {
			try { 
		        System.out.println("\nTestData:");
				this.test_set = readData(test_filepath, test_filepath_labels);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return this.test_set;		
	}
	
	/**
	 * Reads the Mnist files (data and the labels) and stores them into a arrayList of tuples.
	 * @param dataFilePath: The path of the data file
	 * @param labelFilePath: The path of the labels for the data file.
	 * @return An ArrayList of MnistTuple objects that combine the data from the files in an 
	 * easy to read matter.
	 * @throws IOException
	 */
    private ArrayList<MnistMatrix> readData(String dataFilePath, String labelFilePath) throws IOException {

    	// DataInputStream is used b/c it converts directly to primitives.
    	
    	// Getting the pictures
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(
        		new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();
        
        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        // Getting the labels information
        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(
        		new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();
        
        assert numberOfItems == numberOfLabels;

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        ArrayList<MnistMatrix> data = 						// Stores the images and their corresponding outputs
        		new ArrayList<MnistMatrix>(numberOfItems);	// into a list of "tuples" (objects).
        
        // Loops through the file reading data into the array of MnistMatrix objects
        for(int i = 0; i < numberOfItems; i++) {
            MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols); 
            
            // Gets and sets the label from the file.
            mnistMatrix.setLabel(labelInputStream.readUnsignedByte());
            
            // Reads all the grayscale values and stores them in the matrix.
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    mnistMatrix.setValue(r, c, dataInputStream.readUnsignedByte());
                }
            }
            
            data.add(mnistMatrix);
        }
        
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }

    /**
     * This method prints out a sample of the Mnist dataset.
     */
    public void previewMnist(){
    	// Training data:
        printMnistMatrix(this.train_set.get(0));
        
        // Test data:
        printMnistMatrix(this.test_set.get(0));
        
        System.out.print(this.test_set.get(0).getExpectedOut());
    }

    public static void printMnistMatrix(MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());        
        INDArray image = matrix.getINDMatrix();

		for (int r = 0; r < image.rows(); r++) {
			for (int c = 0; c < image.columns(); c++) {
				System.out.print(image.getInt(r,c) + "\t");
			}
			System.out.println();
		}
        
    }
}
