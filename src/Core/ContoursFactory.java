package Core;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class ContoursFactory {
	public List<MatOfPoint> Contours;
	public List<MatOfPoint> ContoursBorder;
    public MatOfPoint2f Contour2f = new MatOfPoint2f();
    public MatOfPoint2f Contour2fBorder = new MatOfPoint2f();
    public Point Center;
    public Point CenterBorder;
    public int maxContourId;
    public int maxContourIdBorder;
    public Mat convexHullFilled;
    public Mat convexHullOutline;
    public RotatedRect minBoundingRect;
    public RotatedRect minBoundingRectBorder;

	public ContoursFactory(Mat segmentation, Mat segmentationBorder)
	{
		// get contours, largest contour id and contour center
        Contours = getContours(segmentation);
        maxContourId = getLargestContour(Contours);
        Contours.get(maxContourId).convertTo(Contour2f, CvType.CV_32FC2);
        Center = getContoursCenter(Contours, maxContourId);
        // get BORDER contours, largest contour id and contour center
        ContoursBorder = getContours(segmentationBorder);
        maxContourIdBorder = getLargestContour(ContoursBorder);
        ContoursBorder.get(maxContourIdBorder).convertTo(Contour2fBorder, CvType.CV_32FC2);
        CenterBorder = getContoursCenter(ContoursBorder, maxContourIdBorder);
        // get convex hull filled & outlined
        convexHullFilled = getContoursConvexHull(segmentation, Contours, true);
        convexHullOutline = getContoursConvexHull(segmentation, Contours, false);
        minBoundingRect = Imgproc.minAreaRect(Contour2f);
        minBoundingRectBorder = Imgproc.minAreaRect(Contour2fBorder);
	}
	
    private List<MatOfPoint> getContours (Mat lesionMask) {
      	 List<MatOfPoint> contours = new ArrayList<>();
         Mat hierarchy = new Mat();
         if (lesionMask.type() != CvType.CV_8UC1) {
        	 Imgproc.cvtColor(lesionMask, lesionMask, Imgproc.COLOR_BGR2GRAY);
         }
         Imgproc.findContours(lesionMask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
      	 return contours;
      }
      
      private int getLargestContour (List<MatOfPoint> contours) {
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
      
      private Point getContoursCenter (List<MatOfPoint> contours, int maxContourId) {
      	 Moments p = Imgproc.moments(contours.get(maxContourId), true);
           int cx = (int) (p.get_m10() / p.get_m00());
           int cy = (int) (p.get_m01() / p.get_m00());
           Point center = new Point(cx,cy);
      	return center;
      }
      
      private Mat getContoursConvexHull (Mat segmentation, List<MatOfPoint> contours, Boolean filled) {
      	List<MatOfPoint> hullList = new ArrayList<>();
      	for (MatOfPoint contour : contours) {
              MatOfInt hull = new MatOfInt();
              Imgproc.convexHull(contour, hull);
              Point[] contourArray = contour.toArray();
              Point[] hullPoints = new Point[hull.rows()];
              List<Integer> hullContourIdxList = hull.toList();
              for (int i = 0; i < hullContourIdxList.size(); i++) {
                  hullPoints[i] = contourArray[hullContourIdxList.get(i)];
              }
              hullList.add(new MatOfPoint(hullPoints));
          }
          Mat convexHull = Mat.zeros(segmentation.size(), segmentation.type());
          for (int i = 0; i < contours.size(); i++) {
        	  if (filled == true) {
        		  Imgproc.drawContours(convexHull, hullList, i, new Scalar(255,255,255), -1);
        	  } else if (filled == false) {
                  Imgproc.drawContours(convexHull, hullList, i, new Scalar(255,255,255), 1);
        	  }
          }

      	return convexHull;
      }    
}
