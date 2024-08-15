package ImageProcessors;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class SegmentationImageProcessorV1 {
	public Mat getMatrix(Mat src)
	{
		// otsu thresholding
	    Mat ots = new Mat();
        Imgproc.cvtColor(src, ots, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(ots, ots, 21);
	    Imgproc.threshold(ots, ots, 0, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
	    
	    // morphological closing
	    Mat morph = new Mat();
	    int kernelSize3 = 7;
        Mat element3 = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_ELLIPSE, new Size(2 * kernelSize3 + 1, 2 * kernelSize3 + 1), new Point(kernelSize3, kernelSize3));
        Imgproc.morphologyEx(ots, morph, Imgproc.MORPH_CLOSE, element3);
        
        // select and draw largest contour
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morph, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
        int maxContourID = obtainLargestContour(contours);
        MatOfPoint2f contour2f = new MatOfPoint2f();
        contours.get(maxContourID).convertTo(contour2f, CvType.CV_32FC2);
        
        Mat lesionMask = Mat.zeros(src.rows(), src.cols(), CvType.CV_8UC1);
        Imgproc.drawContours(lesionMask, contours, maxContourID, new Scalar(255,255,255), -1);
        
    	return lesionMask;
    	
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
