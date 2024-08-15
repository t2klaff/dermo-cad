package ImageProcessors;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import Core.ContoursFactory;

public class ConvexHullImageProcessor {
	public Mat getMatrix(Mat lesionMask, Mat lesionMaskBORDER, ContoursFactory contoursFactory)
	{
        // get convex hull filled and outline
        return obtainContoursConvexHullFilled(lesionMask, contoursFactory.Contours);
	}

   private Mat obtainContoursConvexHullFilled (Mat lesionMask, List<MatOfPoint> contours) {
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
       Mat convexHullFilled = Mat.zeros(lesionMask.size(), lesionMask.type());
       for (int i = 0; i < contours.size(); i++) {
           Imgproc.drawContours(convexHullFilled, hullList, i, new Scalar(255,255,255), -1);
       }
       
   	return convexHullFilled;
   }
}
