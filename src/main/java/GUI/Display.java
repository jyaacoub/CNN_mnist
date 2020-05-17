package GUI;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import ConnectedNetwork.Network;
import ConnectedNetwork.networkMain;
import ConnectedNetwork.networkOperations;
import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.geometry.Rectangle2D;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.effect.ColorAdjust;
import javafx.scene.image.Image;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

/**
 * Main class that provides a GUI for the user to interact with the neural network.
 * @author jyaac
 *
 */
public class Display extends Application {
	
	private static Label instLabel = new Label("Hold up the number to the camera within the boxed region."),
			numbLabel = new Label("That number is..."),
			numbResult = new Label("0");
	
	private static Button displayGrayscalesBtn = new Button("Print gray values");
	
	// Video stream vars:
	private static Canvas mainCanvas = new Canvas(640, 480);
	private static GraphicsContext gc_cam;
	private static AnimationTimer ani_CamView;
	private static VideoCapture vidCap;
	private static ColorAdjust greyscaler =  new ColorAdjust();

	private Stage stage;

	private int netThreshold = 95; 	// Initial threshold to make numbers are more visible
									// (changes depending on lighting)

	protected static int sizeOfRect = 100;
	protected static int sizeOfNetView = 28;
	
	// Convolutional neural network stuff
	private static Network trainedCNN = (new networkOperations()).getNetwork(networkMain.fileName);
	private static Canvas networkCanvas = new Canvas(sizeOfNetView, sizeOfNetView);
	private static GraphicsContext gc_net;
	private static AnimationTimer ani_NetReadAndView;

	@Override
	public void start(Stage arg0) throws Exception {
		this.stage = arg0;
		
		openWebcam(); // builds access to the webcam.
		
		// Initializes graphics contexts tools used to edit the canvases.
		
		// WebCam view canvas editor:
		gc_cam = mainCanvas.getGraphicsContext2D();
		gc_cam.setStroke(Color.GREEN);
		gc_cam.setLineWidth(2);
		
		// Network View canvas editor:
		gc_net = networkCanvas.getGraphicsContext2D();
		
		Group camView = new Group(mainCanvas);
		final Group netView = new Group(networkCanvas);
		
		// Groups the output number and its label.
		VBox networkGuess = new VBox(20, numbLabel, numbResult);
		networkGuess.setAlignment(Pos.CENTER);
		
		//	Groups together all the elements of the right block containing the network View 
		//	and the network's guess. 
		VBox info = new VBox(mainCanvas.getWidth()/4, networkGuess, netView, displayGrayscalesBtn);
		info.setAlignment(Pos.CENTER);
		
		// Groups the camera with the info region centering it.
		HBox camAndInfo = new HBox(20, camView, info);
		camAndInfo.setAlignment(Pos.CENTER);		
		
		// Groups everything above with a direction label to inform the user how to use app.
		VBox root = new VBox(20,instLabel, camAndInfo);
		root.setAlignment(Pos.CENTER);
		
		Scene scene = new Scene(root, 800,560);
		stage = new Stage();
		
		// Centering the stage:
		Rectangle2D screenBounds = Screen.getPrimary().getVisualBounds();
		double width = screenBounds.getWidth();
		double height = screenBounds.getHeight();
		stage.setX((width- scene.getWidth())/2);
		stage.setY((height - scene.getHeight())/2);
		
		stage.setScene(scene);
		stage.setResizable(false);
		stage.show();
		
		// Button to display the exact grayscale values that the network reads:
		displayGrayscalesBtn.setOnAction( new EventHandler<ActionEvent>() {
			Mat m = new Mat();
			
			public void handle(ActionEvent arg0) {
				vidCap.read(m);
				
				int rX_Coor = (m.width()-sizeOfRect)/2, 
						rY_Coor = (m.height()-sizeOfRect)/2;
				
				// Gets the section of the frame that is to be read by net:
				BufferedImage im =  Mat2BufferedImage(m)
						.getSubimage(rX_Coor, rY_Coor, sizeOfRect, sizeOfRect);
				
				// Reduces it to the proper resolution:
				im = resizeImage(im, im.getType());
				
				printGrayValues(im);
			}
			
		});
		
		// Animator to show the cameraView
		ani_CamView = new AnimationTimer() {
			Mat m = new Mat();
			
			@Override
			public void handle(long arg0) {
				// Draws the image from the webcam onto canvas.
				vidCap.read(m);
				gc_cam.drawImage(mat2Image(m), 0, 0); 
				
				
				// Draws a rectangle indicating where the numbers will be read from.
				int rX_Coor = (m.width()-sizeOfRect)/2, 
						rY_Coor = (m.height()-sizeOfRect)/2;
				gc_cam.strokeRect(rX_Coor, rY_Coor,  sizeOfRect, sizeOfRect); 
			}
			
		};
		
		
		// Animator to read numbers and display what the network sees.
		ani_NetReadAndView = new AnimationTimer() {	
			 Mat m = new Mat();
			
			@Override
			public void handle(long now) {
				vidCap.read(m);
				
				int rX_Coor = (m.width()-sizeOfRect)/2, 
						rY_Coor = (m.height()-sizeOfRect)/2;
				
				// Gets the section of the frame that is to be read by net:
				BufferedImage inputImage =  Mat2BufferedImage(m)
						.getSubimage(rX_Coor, rY_Coor, sizeOfRect, sizeOfRect);
				
				// Reduces it to the proper resolution:
				inputImage = resizeImage(inputImage,inputImage.getType());
				
				// Shows user the view of the network:
				Image img = SwingFXUtils.toFXImage(inputImage, null);
				netView.setEffect(greyscaler);								
				gc_net.drawImage(img, 0,0);
				
				numbResult.setText(runThroughNetwork(inputImage).toString()); // Reads and outputs result.
			}
			
		};
		
		ani_NetReadAndView.start();		
		ani_CamView.start();
		
	}
	
	protected static BufferedImage resizeImage(BufferedImage originalImage, int type) {
		BufferedImage resizedImage = new BufferedImage(sizeOfNetView, sizeOfNetView, type);
		
		Graphics2D g = resizedImage.createGraphics();
		g.drawImage(originalImage, 0, 0, sizeOfNetView, sizeOfNetView, null);
		g.dispose();
		
		return resizedImage;
	}
	
    protected void printGrayValues(BufferedImage im) {
		float[] grayData = preprocessData(im);
		System.out.println("Thres: " + netThreshold);
		System.out.println("NetEst: " + runThroughNetwork(im));
		
		
		for (int i = 0; i < 28*28; i++) {
			System.out.printf(" %3d ", (int) grayData[i]);
			
			if ((i + 1)% 28 == 0) { 
				System.out.println();
			}
		}

		System.out.println("\n");
	}

    /**
     * Pre-processing step to ensure data is in the right format before inputting through network.
     * Also makes the numbers stand out more so it is easier for the network
     */
    private float[] preprocessData(BufferedImage im) {
		float[] grayData = new float[sizeOfNetView*sizeOfNetView];
		
		int index = 0;
		int avg = 0;
		
		for (int x=0; x < sizeOfNetView; x++) {
			for (int y=0; y < sizeOfNetView; y++) {
				int gray = (im.getRGB(y,x)& 0xFF);
				gray = 255 - gray; // flips the numbers
				
				avg += gray;
				
				// sets numbers that are not dark enough to 0 (eliminates noise)
				if (gray <= netThreshold) {
					gray = 0; 
				}
				
				grayData[index] = gray;
		    	index++;
		    }
		}
		
		netThreshold = avg/(sizeOfNetView*sizeOfNetView); // updates the threshold to be met.
		
		return grayData;
    }
    
	protected Integer runThroughNetwork(BufferedImage im) {		
		float[] grayData = preprocessData(im);
		
		INDArray input = Nd4j.create(grayData, new int[] {sizeOfNetView*sizeOfNetView,1});
		
		return trainedCNN.getResult(input);
    }

	private void openWebcam() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        vidCap = new VideoCapture();
        vidCap.open(0);

        System.out.println("Camera open: " + vidCap.isOpened());

        stage.setOnCloseRequest(new EventHandler<WindowEvent>() {
            public void handle(WindowEvent we) {

                ani_CamView.stop();
                vidCap.release();

                System.out.println("Camera released");

            }
        });

    }
	
	public static BufferedImage Mat2BufferedImage(Mat m) {
		// source:
		// http://answers.opencv.org/question/10344/opencv-java-load-image-to-gui/
		// Fastest code
		// The output can be assigned either to a BufferedImage or to an Image

		int type = BufferedImage.TYPE_BYTE_GRAY;
		if (m.channels() > 1) { 
			type = BufferedImage.TYPE_3BYTE_BGR; // uses the 3 channel format (e.g.: RGB) for mats with > 1 channel.
		}
		
		int bufferSize = m.channels() * m.cols() * m.rows(); // gets the number of bytes req for the entire mat.
//		System.out.println("size = "+ bufferSize/1000 + " KB");
		byte[] b = new byte[bufferSize];
		m.get(0, 0, b); // get all the pixels
		
		BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(b, 0, targetPixels, 0, b.length);
		return image;

	}
    
    public static Image mat2Image(Mat mat) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", mat, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }
	
	public static void main(String[] args) throws InterruptedException {		
		greyscaler.setSaturation(-1);
		launch(args);
	}

}
