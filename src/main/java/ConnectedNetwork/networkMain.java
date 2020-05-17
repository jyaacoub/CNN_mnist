package ConnectedNetwork;

public class networkMain{
	public static String fileName = "NewNet.ser"; // Net2 > Net1
	
	// And here is where the action happens...
	public static void main(String[] args) {
		networkOperations ops = new networkOperations();
		
		Network n = new Network(new int[] {784,100,10});
		ops.trainTestNetwork(n, 5, 10, 1.5);
		ops.testNetwork(n);
		ops.saveNetwork(n, fileName);
		
//		Network n2 = ops.getNetwork(fileName);
//		ops.testNetwork(n2);
	}
}