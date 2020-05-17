package ConnectedNetwork;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import MNISTReading.MnistDataReader;
import MNISTReading.MnistMatrix;

/**
 * Class use to test and train networks as well as saving and retrieving networks from files.
 * @author jyaac
 */
public class networkOperations{	

	private ArrayList<MnistMatrix> train_set;
	private ArrayList<MnistMatrix> test_set;
	
	public networkOperations(){
		MnistDataReader reader = new MnistDataReader();
		this.train_set = reader.getTrainData();
		this.test_set = reader.getTestData();
	}
	
	/**
	 * Function to get a serialized network 
	 * @param fileName: name of the file that contains the network (file.ser)
	 * @return the network or null if none exists
	 */
	public Network getNetwork(String fileName) {
		Network sub = null;
		try {
			FileInputStream file = new FileInputStream(fileName);
			ObjectInputStream in = new ObjectInputStream(file);
			
			sub = (Network)in.readObject();
			
			file.close();
			in.close();
			System.out.println("Network successfully retrived...");			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return sub;
	}
	
	/**
	 * Saves a network to be used later
	 * @param net: The Network object to be serialized
	 * @param fileName: the name of the file to save to (creates a new file if none exists)
	 */
	public void saveNetwork(Network net, String fileName) {
		try {
			FileOutputStream file = new FileOutputStream(fileName, false);
			ObjectOutputStream out = new ObjectOutputStream(file);
			
			out.writeObject(net);
			
			out.close();
			file.close();
			System.out.println("Network saved...");			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Runs the network through the Mnist numbers test set to determine accuracy
	 * @param net: the network to test
	 */
	public void testNetwork(Network net) {
		System.out.printf("Accuracy: %s/%s\n", (net).evaluate(this.test_set), this.test_set.size());
	}

	public void trainNetwork(Network net, int epochs, int batch_size, double eta) {
		net.SGD(this.train_set, epochs, batch_size, eta);
	}

	public void trainTestNetwork(Network net, int epochs, int batch_size, double eta) {
		net.SGD(this.train_set, epochs, batch_size, eta, this.test_set);		
	}
}
