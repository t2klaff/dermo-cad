package Features;

import java.util.List;

import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class CircularityFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getCircularity(contoursFactory.Contours, contoursFactory.maxContourId);
	}
	
    private Double getCircularity (List<MatOfPoint> contours, int maxContourId) {
   	 	Moments p = Imgproc.moments(contours.get(maxContourId), true);
   	 	double m00 = p.get_m00();
   	 	double m20 = p.get_mu20();
   	 	double m02 = p.get_mu02();
   	 	
    	double huCircularity = (Math.pow(m00, 2)) / ((2*Math.PI)*(m20+m02));

    	return huCircularity;
    }
}
