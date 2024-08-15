package ImageProcessors;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class SegmentationImageProcessorV2 {
	
	public Mat getMatrix(Mat src)
	{
		Mat lab = new Mat();
        Imgproc.cvtColor(src, lab, Imgproc.COLOR_BGR2Lab);
		
        // split bgr channels
		List<Mat> dstBGR = splitThresholdBGR(src);
 	    Mat candidate1 = dstBGR.get(0);
//        HighGui.imshow( "c1bgr", candidate1 );
//        HighGui.imshow( "c2bgr", dstBGR.get(1) );
//        HighGui.imshow( "c3bgr", dstBGR.get(2) );

 	    // split lab channels
        List<Mat> dstLAB = splitThresholdLAB(lab);
	    Mat candidate2 = dstLAB.get(2);
//        HighGui.imshow( "c1lab", candidate2 );
//        HighGui.imshow( "c2lab", dstLAB.get(1) );
//        HighGui.imshow( "c3lab", dstLAB.get(2) );

	    List<MatOfPoint> contours1 = obtainContours(candidate1);
        List<MatOfPoint> contours2 = obtainContours(candidate2);
        int maxContourID1 = obtainLargestContour(contours1);
        int maxContourID2 = obtainLargestContour(contours2);
        
        boolean accept1 = checkContourAtBorder(contours1.get(maxContourID1));
        boolean accept2 = checkContourAtBorder(contours2.get(maxContourID2));
        
        Mat zeros1 = Mat.zeros(src.size(), CvType.CV_8UC1);
        Mat zeros2 = Mat.zeros(src.size(), CvType.CV_8UC1);
        Imgproc.drawContours(zeros1, contours1, maxContourID1, new Scalar(255, 255, 255), -1);
        Imgproc.drawContours(zeros2, contours2, maxContourID2, new Scalar(255, 255, 255), -1);
        
//        Mat i1 = src.clone();
//        Mat i2 = src.clone();
//        Imgproc.drawContours(i1, contours1, maxContourID1, new Scalar(255, 0, 255), 2);
//        Imgproc.drawContours(i2, contours2, maxContourID2, new Scalar(255, 255, 0), 2);
//        
//        HighGui.imshow( "candidate1", i1 );
//        HighGui.imshow( "candidate2", i2 );
//        HighGui.waitKey(0);
        
        Mat result = new Mat();
        
        // if both segmentations pass checks, combine them
        // if only one passes, use only that one
        // if neither pass, combine them and warn of bad segmentation
        if (accept1 == true && accept2 == true) {
//        	System.out.println("both segmentations are legal, union before morphology...");
            result = Mat.zeros(src.size(), CvType.CV_8UC1);
            Core.bitwise_or(zeros1, zeros2, result);
        } else if (accept1 == true && accept2 == false) {
//        	System.out.println("BGR[b] is legal, proceeding to morphology");
        	result = zeros1;
        } else if (accept1 == false && accept2 == true) {
//        	System.out.println("LAB[b] is legal, proceeding to morphology");
        	result = zeros2;
        } else {
        	// no segmentation found
//        	System.out.println("ERROR: no legal segmentation found! WARNING: proceeding with bad segmentation");
        	result = Mat.zeros(src.size(), CvType.CV_8UC1);
            Core.bitwise_or(zeros1, zeros2, result);
        }
        
        // refinement morphology
        Mat morph = Mat.zeros(src.size(), CvType.CV_8UC1);
        int kernelSize = 5;
        Mat element = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_ELLIPSE, new Size(2 * kernelSize + 1, 2 * kernelSize + 1), new Point(kernelSize, kernelSize));
        Imgproc.morphologyEx(result, morph, Imgproc.MORPH_CLOSE, element);
        Imgproc.morphologyEx(morph, morph, Imgproc.MORPH_CLOSE, element);
        Mat segmentation = morph.clone();
        
//        HighGui.waitKey();
        
    	return segmentation;
	}
	
	private List<MatOfPoint> obtainContours(Mat binMask) {
    	List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binMask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
   	 	return contours;
    }
	
    private int obtainLargestContour (List<MatOfPoint> contours) {
        double maxVal = 0;
        int maxContourID = 0;
        for (int contourID = 0; contourID < contours.size(); contourID++)
        {
            double contourArea = Imgproc.contourArea(contours.get(contourID));
            if (maxVal < contourArea) {
                maxVal = contourArea;
                maxContourID = contourID;
            }
        }
    	return maxContourID;
    }
    
    private boolean checkContourAtBorder(MatOfPoint contour) {
    	boolean accept = true;
        Point[] points = contour.toArray();
        BitSet bs = new BitSet(4);
        for (Point p : points) {
        	if (p.x == 0) {
        		bs.set(0);
        	}
        	if (p.x == 599) {
        		bs.set(1);
        	}
        	if (p.y == 0) {
        		bs.set(2);
        	}
        	if (p.y == 449) {
        		bs.set(3);
        	}
        }
        int hitCount = bs.cardinality();
        if (hitCount > 1) {
        	accept = false;
        }
    	return accept;
    }
    
    private List<Mat> splitThresholdBGR(Mat src) {
    	List<Mat> bgrMats = new ArrayList<>();
        
    	Mat blur = src.clone();
        Imgproc.medianBlur(src, blur, 21);
    	
        Mat c1 = new Mat();
        Core.extractChannel(blur, c1, 0);
    	Mat thrC1 = new Mat();
//        Imgproc.cvtColor(src, thrR, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(c1, thrC1, 9);
	    Imgproc.threshold(thrC1, thrC1, 50, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
	    bgrMats.add(thrC1);
        
        Mat c2 = new Mat();
        Core.extractChannel(blur, c2, 1);
    	Mat thrC2 = new Mat();
//        Imgproc.cvtColor(src, thrR, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(c2, thrC2, 9);
	    Imgproc.threshold(thrC2, thrC2, 50, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
	    bgrMats.add(thrC2);
        
        Mat c3 = new Mat();
        Core.extractChannel(blur, c3, 2);
    	Mat thrC3 = new Mat();
//        Imgproc.cvtColor(src, thrR, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(c3, thrC3, 9);
	    Imgproc.threshold(thrC3, thrC3, 50, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
	    bgrMats.add(thrC3);
        
    	return bgrMats;
    }
    
    private List<Mat> splitThresholdLAB(Mat src) {
    	List<Mat> labMats = new ArrayList<>();
        
    	Mat blur = src.clone();
        Imgproc.medianBlur(src, blur, 21);
    	
        Mat c1 = new Mat();
        Core.extractChannel(blur, c1, 0);
    	Mat thrC1 = new Mat();
//        Imgproc.cvtColor(src, thrR, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(c1, thrC1, 9);
	    Imgproc.threshold(thrC1, thrC1, 50, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
	    labMats.add(thrC1);
        
        Mat c2 = new Mat();
        Core.extractChannel(blur, c2, 1);
    	Mat thrC2 = new Mat();
//        Imgproc.cvtColor(src, thrR, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(c2, thrC2, 9);
	    Imgproc.threshold(thrC2, thrC2, 50, 255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
	    labMats.add(thrC2);
        
        Mat c3 = new Mat();
        Core.extractChannel(blur, c3, 2);
    	Mat thrC3 = new Mat();
//        Imgproc.cvtColor(src, thrR, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(c3, thrC3, 9);
	    Imgproc.threshold(thrC3, thrC3, 50, 255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
	    if (thrC3.get(0,0)[0] == 255 &&  thrC3.get(0,599)[0] == 255 && thrC3.get(449,0)[0] == 255 && thrC3.get(449,599)[0] == 255) {
	    	Core.bitwise_not(thrC3, thrC3);
	    }
	    labMats.add(thrC3);

    	return labMats;
    }
	
	
}