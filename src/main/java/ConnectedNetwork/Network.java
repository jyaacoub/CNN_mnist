package ConnectedNetwork;

import MNISTReading.*;
import java.util.ArrayList;
import java.util.Collections;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Network implements java.io.Serializable{
	
	private static final long serialVersionUID = 219L;
	
	int num_layers;
	int[] layerSizes;
	ArrayList<INDArray> biases = new ArrayList<INDArray>(); // list of the bias vectors connecting each layer.
	ArrayList<INDArray> weights = new ArrayList<INDArray>(); // List of matrices that holds the weights connecting each layer
	
	/**
	 * Creates a untrained neural network with a specific structure.
	 *  @param sizes: the sizes of each layer in the network including output and input layers.
	 */ 
	Network(int[] sizes){
		this.num_layers = sizes.length;
		this.layerSizes = sizes;
		
		// Now we RANDOMLY select biases and weights for our network:
		
		// Biases selection:
		// First layer doesn't have biases so it is ignored (i=1)
		for (int i = 1; i < num_layers; i++) {
			INDArray vector = Nd4j.randn(layerSizes[i],1); // Fills each column with gaussian random numbers (mean of 0 with SD of 1)
			biases.add(vector);
		}
		
		// Weights selection:
		for (int i = 0; i < num_layers-1; i++) {
			INDArray matrix = Nd4j.randn(sizes[i+1], sizes[i]);	// A matrix that is L' x L. 
			weights.add(matrix);								// Thereby accommodating all the weights connecting L and L' layers.
		}
	}

	/**
	 * Returns the output vector of the network.
	 * @param in: the INDArray vector of grayscale values from the image.
	 * @return A INDArray output vector.
	 */
	public INDArray feedForward(INDArray in) {
		for (int i = 0; i < num_layers-1; i++) {	
			in = sigmoid(
					weights.get(i).mmul(in)		// w*a
					.add(biases.get(i)) ); 		// + b
		}
		return in;
	}
	
	/**
	 * Returns the exact digit result when passing image through network.
	 * @param in: the INDArray vector of grayscale values from the image.
	 * @return A number from 0-9 that is the output of the network.
	 */
	public int getResult(INDArray in) {
		int result = feedForward(in)
				.argMax() 	// returns a INDArray of the index that has the highest activation of the "matrix"
				.getInt(0); // returns a int of that index.
		return result;		
	}

	// Returns number of right outputs to a given test data, to determine network accuracy.
	public int evaluate(ArrayList<MnistMatrix> test_data) {
		int totalCorrect = 0;
		
		for (MnistMatrix tuple : test_data) {
			int obs = getResult(tuple.getINDArray());
			int exp = tuple.getLabel();
			
			if (obs == exp) {
				totalCorrect++;
			}	
		}
		
		return totalCorrect;
	}
	
	/**
	 * Trains the network by stochastic gradient descent. With test data to track its progress through each epoch.
	 * @param training_data: A list of tuples containing the input (the image as INDArray) and the correct output (an int from 0-9).
	 * @param epochs: the number of epochs of training to be done.
	 * @param mini_batch_size: the size of the mini batches.
	 * @param eta: the learning rate.
	 * @param test_data: the test data to determine accuracy of network after each epoch.
	 */
	public void SGD(ArrayList<MnistMatrix> training_data, int epochs, int mini_batch_size, double eta,
			ArrayList<MnistMatrix> test_data) {
		// Stores sizes of the input data
		int test_data_size = test_data.size();
		int train_data_size = training_data.size();
		
		int eval = evaluate(test_data);
		
		System.out.printf("Before Training: %s/%s = %.4f\n", eval, test_data_size, (float) eval/ (float)test_data_size);
		
		// Runs j many epochs.
		// This process is similar to n-fold cross validation but with training.
		for (int j = 0 ; j < epochs ; j++) {
			Collections.shuffle(training_data);	// shuffles the training data
			
			// Creates mini-batches from shuffled data with the given size
			ArrayList<ArrayList<MnistMatrix>> mini_batches = new ArrayList<ArrayList<MnistMatrix>>();
			
			for (int k = 0 ; k + mini_batch_size < train_data_size ; k += mini_batch_size) {
				ArrayList<MnistMatrix> in = new ArrayList<MnistMatrix>(training_data.subList(k, k + mini_batch_size));
				mini_batches.add(in);
			}
			// We don't include any remaining data so that all batches are the same size
			
			// Executes and learns from the outputs of each mini batch, updating along the way.
			for (ArrayList<MnistMatrix> mini_batch : mini_batches) {
				update_mini_batch(mini_batch, eta);
			}
			
			int eval_ep = evaluate(test_data);
			
			System.out.printf("Epoch %s: %s/%s = %.4f\n", j, eval_ep, test_data_size, (float)eval_ep / (float)test_data_size);	
		}
	}
	
	/**
	 * Trains the network by stochastic gradient descent. WITHOUT EVALUATIONS ON EACH EPOCH
	 * @param training_data: A list of tuples containing the input (the image as INDArray 8x8) and the correct output (an int from 0-9).
	 * @param epochs: the number of epochs of training to be done.
	 * @param mini_batch_size: the size of the mini batches.
	 * @param eta: the learning rate.
	 */
	public void SGD(ArrayList<MnistMatrix> training_data, int epochs, int mini_batch_size, double eta) {
		int train_data_size = training_data.size();
		
		// Runs j many epochs.
		// This process is similar to n-fold cross validation but with training.
		for (int j = 0 ; j < epochs ; j++) {
			Collections.shuffle(training_data); // shuffles the training data
			
			// Creates mini-batches from shuffled data with the given size
			ArrayList<ArrayList<MnistMatrix>> mini_batches = new ArrayList<ArrayList<MnistMatrix>>();
			
			for (int k = 0 ; k+mini_batch_size < train_data_size; k += mini_batch_size) {
				ArrayList<MnistMatrix> in = new ArrayList<MnistMatrix>(training_data.subList(k, k+mini_batch_size));
				mini_batches.add(in);
			}
			// We don't include any remaining data so that all batches are the same size
			
			// Executes and learns from the outputs of each mini batch, updating along the way.
			for (ArrayList<MnistMatrix> mini_batch : mini_batches) {
				update_mini_batch(mini_batch, eta);
			}
			
			System.out.printf("Epoch %s complete \n", j);	// NO EVALUATION (faster)
		}
	}
	
	// Using backpropagation with gradient descent on the batch to update the weights and biases.
	private void update_mini_batch(ArrayList<MnistMatrix> mini_batch, double eta) {
		// Initializes and builds the nablas to be filled with backprop. data.
		ArrayList<INDArray> nabla_b = new ArrayList<INDArray>(); // holds the biases for each layer
		for (INDArray b: this.biases) {
			nabla_b.add(Nd4j.zeros(b.shape()));
		}
		
		ArrayList<INDArray> nabla_w = new ArrayList<INDArray>(); // holds the weights for each layer
		for (INDArray w: this.weights) {
			nabla_w.add(Nd4j.zeros(w.shape()));
		}
		
		// Loops through the mini_batch to get the summation for all the individual
		// nablas for each test data to estimate the true gradient.
		for (MnistMatrix matrix : mini_batch) {
			ArrayList<ArrayList<INDArray>> delta_nabla = backprop(matrix); // returns (b,w) tuple (updates to perform)
			ArrayList<INDArray> delta_nabla_b = delta_nabla.get(0);
			ArrayList<INDArray> delta_nabla_w = delta_nabla.get(1);
			
			assert delta_nabla_b.size()== nabla_b.size();
			assert delta_nabla_w.size()== nabla_w.size();
			
			// Sums the changes to the nablas
			for (int i = 0; i < delta_nabla_b.size(); i++) {
				nabla_b.get(i).addi(delta_nabla_b.get(i));
			}
			for (int i = 0; i < delta_nabla_w.size(); i++) {
				nabla_w.get(i).addi(delta_nabla_w.get(i));
			}
		}
		
		assert this.biases.size() == nabla_b.size();
		assert this.weights.size() == nabla_w.size();
		
		// Updates the networks weights and biases from the calculated gradient of the cost function:
		
		// Updates weights
		for (int i = 0 ; i < nabla_w.size(); i++) {
			this.weights.get(i).subi(										// w - 
							nabla_w.get(i).mul(eta / mini_batch.size()) );	// w'*(eta/len(mini_batch))
		}
		
		// Updates biases
		for (int i = 0; i < nabla_b.size(); i++) {			
			this.biases.get(i).subi(										// b -
							nabla_b.get(i).mul(eta / mini_batch.size()) ); 	// b'*(eta/len(mini_batch))
		}
	}

	// Returns a (b,w) tuple representing the changes for the biases and weights respectfully
	// which is the gradient of the cost function.
	private ArrayList<ArrayList<INDArray>> backprop(MnistMatrix matrix) {
		ArrayList<INDArray> nabla_b = new ArrayList<INDArray>(); // holds the biases for each layer
		for (INDArray b: this.biases) {
			nabla_b.add(Nd4j.zeros(b.shape()));
		}
		
		assert nabla_b.get(0).dataType().toString().equals("FLOAT");
		
		ArrayList<INDArray> nabla_w = new ArrayList<INDArray>(); // holds the weights for each layer
		for (INDArray w: this.weights) {
			nabla_w.add(Nd4j.zeros(w.shape()));
		}
		
		INDArray activation = matrix.getINDArray();
		ArrayList<INDArray> activations = new ArrayList<INDArray>();
		activations.add(activation); 	// Input activation is simply the grayscale values of the image itself.
		
		ArrayList<INDArray> Z_Vectors = new ArrayList<INDArray>(); // stores all the z vectors for each layer (w*a + b).
		
		// We don't utilize the "feedForward" method b/c we need to store all the individual z vectors and the activations:
		for (int i = 0; i < num_layers-1; i++ ) {
			INDArray w = this.weights.get(i),
					b = this.biases.get(i);
			INDArray z = ( w.mmul(activation) ).add(b); // feeds forward

			Z_Vectors.add(z);
			activation = sigmoid(z);
			activations.add(activation);			
		}
		
		// Backward pass:
		INDArray delta = quadCostDelta(activations.get(activations.size()-1), matrix.getExpectedOut(), Z_Vectors);	// Computing the output error
		
		INDArray activationT = activations.get(activations.size()-2).transpose(); // Allows for formation of w matrix at L.
		
		nabla_b.set(nabla_b.size()-1, delta);											// The error is related to the gradients by
		nabla_w.set(nabla_w.size()-1, 													// nabla_b = error
				delta.mmul(activationT) );												// nabla_w = activation(L-1) * error
		
		// The following utilizes BP2 to backprop the error and retrieve the rest of the gradients
		// to build the gradient vectors for the cost function.
		for (int L = 1; L <= num_layers-2; L++) {
			INDArray z = Z_Vectors.get(Z_Vectors.size()-L-1);
			INDArray sp = sigmoidPrime(z);
			INDArray w = weights.get(weights.size()-L).transpose();
			
			delta = ( w.mmul(delta) ).mul(sp); 			// Backprop step: delta_new = ((w_old)^T *delta_old) x sp(z_new)  
														// Where "x" is the hadamard product.
			
			activationT = activations.get(activations.size()-L-2).transpose(); // (a)^T

			nabla_b.set(nabla_b.size()-L-1, delta);
			nabla_w.set(nabla_w.size()-L-1,
					delta.mmul(activationT));			//w' = (a)^T * delta
		}
		
		ArrayList<ArrayList<INDArray>> gradientVect = new ArrayList<ArrayList<INDArray>>();
		gradientVect.add(nabla_b);
		gradientVect.add(nabla_w);
		
		return gradientVect;
	}

	
	/**		Misc. methods: 		*/
	
	// Derivative of 1/2(e-a)^2. (quadratic cost)
	private INDArray quadCostDelta(INDArray out, INDArray expectedOut, ArrayList<INDArray> Z_Vectors) {
		
		return out.sub(expectedOut)											// delta = nabla_a_C x sigmoid'(Z)
				.mul( sigmoidPrime( Z_Vectors.get(Z_Vectors.size()-1) ) );	// Where "x" is a hadamard product.
	}
	
	private INDArray sigmoid(INDArray indArray) {
		return Transforms.sigmoid(indArray, false);
	}
	
	private INDArray sigmoidPrime(INDArray indArray) {
		return Transforms.sigmoidDerivative(indArray, false);
	}
}