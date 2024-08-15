package ImageProcessors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class BlackBordersImageProcessor {
	public Mat getMatrix(Mat src)
	{
		Mat result = new Mat();
//		System.out.println(src.size());
		
		// define regions in corners for checking for black borders
		List<Rect> rois = Arrays.asList(new Rect(0, 0, 25, 25), new Rect(src.width()-25, 0, 25, 25), new Rect(0, src.height()-25, 25, 25), new Rect(src.width()-25, src.height()-25, 25, 25));

		List<Double> rectMeans = new ArrayList<>();
		Boolean hasBlackBorders = false;
		
		// check if any box has very low intensity - probably due to black border
		for (Rect r : rois) {
			Double mMean = 0.0;
			Mat sub = src.submat(r);
			Scalar mean = Core.mean(sub);
			mMean = (mean.val[0] + mean.val[1] + mean.val[2]) / 3;
			rectMeans.add(mMean);
//			Imgproc.rectangle(test, r, new Scalar(0,0,255), 2);
			if (mMean < 15) {
				hasBlackBorders = true;
			}
		}

		// if black borders then remove them
		if (hasBlackBorders == true) {
//			System.out.println("BLACK BORDERS PRESENT");
	    	result = blackBorderRemover(src);
		} else {
//			System.out.println("NO BLACK BORDERS");
			result = src;
		}
        
    	return result;
    	
	}
	
	private Mat blackBorderRemover (Mat src) {
	    Mat gsc = new Mat();
	    Imgproc.cvtColor(src, gsc, Imgproc.COLOR_BGR2GRAY);
	    
	    // blur to help define black border
	    Mat blur = new Mat();
	    Imgproc.medianBlur(gsc, blur, 69);
	    
	    // threshold the image for circle detection
	    Mat thr = new Mat();
	    Imgproc.threshold(blur, thr, 50, 255, Imgproc.THRESH_BINARY);
	    List<MatOfPoint> thrCont = getContours(thr);
	    int maxID = obtainLargestContour(thrCont);
//	    Imgproc.drawContours(inpainted, thrCont, -1, new Scalar(255,0,255), 2);
	    
	    // detect circles using Hough circles
	    Mat circles = new Mat();
        Imgproc.HoughCircles(thr, circles, Imgproc.HOUGH_GRADIENT, 1.0, (double)thr.rows()/2, 30, 15, 250, 2500);
        
        Point center = new Point();
        int radiusOuter = 0;
        int radiusInner = 0;
        
        Mat firstMask = Mat.zeros(gsc.size(), gsc.type());
        Mat result = src.clone();
        
        // if there is circle detected using Hough circles
        if (!circles.empty()) {
        	// get the circle properties such as center and radius
        	for (int x = 0; x < circles.cols(); x++) {
                double[] c = circles.get(0, x);
                center = new Point(Math.round(c[0]), Math.round(c[1]));
                radiusOuter = (int) Math.round(c[2]) - 50;
                radiusInner = radiusOuter-25;
                Imgproc.circle(firstMask, center, radiusOuter, new Scalar(255,255,255), -1);
                Imgproc.circle(firstMask, center, radiusInner, new Scalar(0,0,0), -1);
            }
        // if no circle detected using Hough circles, use standard circle with center at center of image and radius half of image height
        } else if (circles.empty()) {
        	center = new Point(gsc.width()/2, gsc.height()/2);
        	radiusOuter = gsc.height()/2;
        	radiusInner = radiusOuter-25;
        	Imgproc.circle(firstMask, center, radiusOuter, new Scalar(255,255,255), -1);
        	Imgproc.circle(firstMask, center, radiusInner, new Scalar(0,0,0), -1);
        }    

        List<Mat> circleMasks = new ArrayList<>();
        
        //make up to 25 circles increasing size
        for (int r=0; r<25; r++) {
            Mat mask = Mat.zeros(gsc.size(), gsc.type());
            Imgproc.circle(mask, center, radiusOuter, new Scalar(255,255,255), -1);
            Imgproc.circle(mask, center, radiusInner, new Scalar(0,0,0), -1);
            radiusOuter += 25;
            radiusInner += 25;
        	circleMasks.add(mask);
        }

    	Double curBMean = 0.0;
    	Double curGMean = 0.0;
    	Double curRMean = 0.0;
    	Double curMeanMean = 0.0;
        int iterCount = 1;
        
        // for all the concentric circles of increasing size
        for (Mat m : circleMasks) {
//            System.out.println(iterCount);
            int nonZeroCount = Core.countNonZero(m);
            
            // mask the regon of interest
            Mat maskClone = m.clone();
            Imgproc.cvtColor(m, maskClone, Imgproc.COLOR_GRAY2BGR);
            Mat masked = new Mat();
            Core.bitwise_and(result, maskClone, masked);
            int rows = masked.rows(); //Calculates number of rows
            int cols = masked.cols(); //Calculates number of columns
            int ch = masked.channels(); //Calculates number of channels (Grayscale: 1, RGB: 3, etc.)
        	
            Double bSum = 0.0;
            Double gSum = 0.0;
            Double rSum = 0.0;
        	
            // calculate average intensities for channels
            for (int i=0; i<rows; i++) {
            	for (int j=0; j<cols; j++) {

            		if (m.get(i,j)[0] > 0) {
//            			Imgproc.circle(masked, new Point(j,i), 1, new Scalar(255,0,255));
                		double[] data = masked.get(i, j);
                		bSum += data[0];
                		gSum += data[1];
                		rSum += data[2];
            		}
            	}
            }
            
            Double bMean = bSum / nonZeroCount;
            Double gMean = gSum / nonZeroCount;
            Double rMean = rSum / nonZeroCount;
            Double meanMean = (bMean + gMean + rMean) / 3;
            
            // if mean is 0 or lower than last
            if (meanMean.isNaN()) {
            	break;
            } else if (meanMean < curMeanMean) {
//            	System.out.println("running adjustments");
            	// run adjustments
            	for (int i=0; i<rows; i++) {
                	for (int j=0; j<cols; j++) {
                		if (m.get(i,j)[0] > 0) {
                			double[] newData = {curBMean, curGMean, curRMean};
                			result.put(i, j, newData);
                		}
                	}
                }
            } else {
            	curBMean = bMean;
            	curGMean = gMean;
            	curRMean = rMean;
            	curMeanMean = meanMean;
            }

//            System.out.println(bMean);
//            System.out.println(gMean);
//            System.out.println(rMean);
        	
            iterCount++;
        }
		return result;
	}
	
    private List<MatOfPoint> getContours (Mat mask) {
     	 List<MatOfPoint> contours = new ArrayList<>();
          Mat hierarchy = new Mat();
          Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
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
}
