package Features;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class RadialVarianceFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getRadialVariance(imagesFactory, imagesFactory.GetMatrix(ImagesFactory.SegmentationV3), imagesFactory.GetMatrix(ImagesFactory.SegmentationBorder), imagesFactory.GetMatrix(ImagesFactory.SegmentationOutlineBorder), contoursFactory.CenterBorder);
	}
	
	private double getRadialVariance (ImagesFactory imagesFactory, Mat Segmentation, Mat SegmentationBorder, Mat SegmentationOutlineBorder, Point center) { 
    	
        int addHeight = ((int)SegmentationOutlineBorder.height() - Segmentation.rows())/2;
        int addWidth = ((int)SegmentationOutlineBorder.width() - Segmentation.cols())/2;
        Rect ogROI = new Rect(addWidth, addHeight, Segmentation.cols(), Segmentation.rows());
		
        List<Point> perimeterPoints = obtainOutlineCoords(SegmentationOutlineBorder);
        
        List<Double> dists = new ArrayList<>(perimeterPoints.size());
        double distsSum = 0;
        for (int k = 0; k < perimeterPoints.size(); k++) {
        	Double dist = Math.sqrt(Math.pow((center.x-perimeterPoints.get(k).x),2)+Math.pow((center.y-perimeterPoints.get(k).y),2));
        	dists.add(dist);
        	distsSum += dist;
        }
        double avgDist = distsSum / perimeterPoints.size();
        Mat circle = Mat.zeros(SegmentationOutlineBorder.size(), SegmentationBorder.type());
        Imgproc.circle(circle, center, (int)avgDist, new Scalar(255,255,255), -1);
        Mat intersection = new Mat();
        Core.bitwise_and(SegmentationBorder, circle, intersection);
        Mat union = new Mat();
        Core.bitwise_or(SegmentationBorder, circle, union);
        double count_white_intersection= Core.countNonZero(intersection);
        double count_white_union= Core.countNonZero(union);
        double intersection_over_union = (count_white_intersection / count_white_union);
//        double intersection_over_union = getIntersectionOverUnion(lesionMask, circle);
        
    	Mat working = SegmentationOutlineBorder.clone();
    	Imgproc.cvtColor(working, working, Imgproc.COLOR_GRAY2BGR);
        Imgproc.drawMarker(working, center, new Scalar(255,49,0,255), Imgproc.MARKER_CROSS, 10, 1);
        Imgproc.circle(working, center, (int)avgDist, new Scalar(255,0,255), 1);
        Mat cropped = new Mat(working, ogROI);
        imagesFactory.AddMatrix("radialVariance", cropped.clone());
//    	HighGui.imshow("radial variance", cropped);
        
        return 1/intersection_over_union;
    }
	
    private List<Point> obtainOutlineCoords (Mat SegmentationOutline) {
        List<Point> outlineCoords = new ArrayList<>();
        Mat nonZeroCoordinates = Mat.zeros(SegmentationOutline.size(), SegmentationOutline.type());
        Core.findNonZero(SegmentationOutline, nonZeroCoordinates);
        MatOfPoint mop = new MatOfPoint(nonZeroCoordinates);
    	for (Point pnt: mop.toList()) {
    		outlineCoords.add(pnt);
    	}
    	return outlineCoords;
    }
}
