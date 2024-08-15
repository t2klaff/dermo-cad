

package ImageProcessors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

public class SegmentationImageProcessorV3 {
	
	public Mat getMatrix(Mat src)
	{
//		System.out.println(src.size());
		 
		Mat gsc = new Mat();
		Imgproc.cvtColor(src.clone(), gsc, Imgproc.COLOR_BGR2GRAY);	
		
		Mat lab = new Mat();
		Imgproc.cvtColor(src.clone(), lab, Imgproc.COLOR_BGR2Lab);
		
		List<MatOfPoint> bgrContours = new ArrayList<>();
		List<MatOfPoint> labContours = new ArrayList<>();
		List<MatOfPoint> bgrClustContours = new ArrayList<>();
		List<MatOfPoint> labClustContours = new ArrayList<>();
		
		// get all bgr threshold segmentation candidates
		for (Mat m : channelSplitterAndThresholder(src, 45)) {
			for (MatOfPoint c : obtainContoursNotAtBorder(getContours(m), m)) {
				bgrContours.add(c);
			}
		}
		
		// get all lab threshold segmentation candidates
		for (Mat m : channelSplitterAndThresholder(lab, 45)) {
			for (MatOfPoint c : obtainContoursNotAtBorder(getContours(m), m)) {
				labContours.add(c);
			}
		}

		// get all colour clustering segmentation candidates
		Mat dst = src.clone();
		Mat blur = new Mat();
        Imgproc.medianBlur(lab, blur, 45);
		List<Mat> clusters = cluster(blur, 7);
		for (Mat m : clusters) {
			Mat cvt = new Mat();
			Imgproc.cvtColor(m, cvt, Imgproc.COLOR_BGR2GRAY);
			List<MatOfPoint> cnts = obtainContoursNotAtBorder(getContours(cvt), cvt);
			for (MatOfPoint c : cnts) {
				labClustContours.add(c);
			}
	        Imgproc.drawContours(dst, cnts, -1, new Scalar(getRand(0,255),getRand(0,255),getRand(0,255)), 2);
		}

		// remove any candidates that are duplicates
		for (int i = 0; i < labClustContours.size(); i++) {
			for (int j = i+1; j < labClustContours.size(); j++) {
				Double r = Imgproc.matchShapes(labClustContours.get(i), labClustContours.get(j), Imgproc.TM_CCORR_NORMED, 0);
//				System.out.println(r);
				if (r < 0.01) {
					labClustContours.remove(j);
				}
			}
		}
		
		List<MatOfPoint> allCandidates = Stream.of(bgrContours, labContours, bgrClustContours, labClustContours).flatMap(Collection::stream).collect(Collectors.toList());
		
		Double max = 1.0;
		int bestLabID = 0;
		int bestBGRID = 0;
		
		// find the best matching candidates as they probably correlate to actual region of interest
		for (int i = 0; i < allCandidates.size(); i++) {
			for (int j = i+1; j < allCandidates.size(); j++) {
				Double r = Imgproc.matchShapes(allCandidates.get(i), allCandidates.get(j), Imgproc.TM_CCORR_NORMED, 0);
				if (r < max) {
					max = r;
					bestLabID = i;
					bestBGRID = j;
				}
			}
		}
		
		Mat zero1 = Mat.zeros(src.size(), CvType.CV_8UC1);
		Mat zero2 = Mat.zeros(src.size(), CvType.CV_8UC1);
        Imgproc.drawContours(zero1, allCandidates, bestLabID, new Scalar(255, 255, 255), -1);
        Imgproc.drawContours(zero2, allCandidates, bestBGRID, new Scalar(255, 255, 255), -1);
    	Mat result = Mat.zeros(src.size(), CvType.CV_8UC1);
        Core.bitwise_or(zero1, zero2, result);

    	return result;
    	
	}
	
	public int getRand(int min, int max) {
	    Random random = new Random();
	    return random.ints(min, max).findFirst().getAsInt();
	}
	
	public Mat getClustered (Mat src, int kSize, int cNum) {
		
//		List<Scalar> cullers = Arrays.asList(new Scalar(1,54,20), new Scalar(1,67,26), new Scalar(1,85,36), new Scalar(34,141,83), new Scalar(66,169,115), new Scalar(118,213,170));
//		List<Scalar> cullers5 = Arrays.asList(new Scalar(210,234,151), new Scalar(161,199,140), new Scalar(148,110,129), new Scalar(108,34,116), new Scalar(66,33,75));
		
		Mat blur = new Mat();
        Imgproc.medianBlur(src, blur, kSize);
		List<Mat> clusters = cluster(blur, cNum);
		List<Mat> clustersDs = new ArrayList<>();
		Mat dst = src.clone();
//		int culIter = 0;
		for (Mat m : clusters) {
			Mat cvt = new Mat();
			Imgproc.cvtColor(m, cvt, Imgproc.COLOR_BGR2GRAY);
			
			List<MatOfPoint> cont = obtainContoursNotAtBorder(getContours(cvt), src);
			Mat r = new Mat();
			Imgproc.resize(m, r, new Size(m.width()/2, m.height()/2));
			Imgproc.resize(cvt, cvt, new Size(m.width()/2, m.height()/2));
			clustersDs.add(cvt);
//			culIter++;
		}
		Mat collage = new Mat();
		Core.hconcat(clustersDs, collage);
		return collage;
	}
	
    private List<MatOfPoint> getContours (Mat mask) {
    	 List<MatOfPoint> contours = new ArrayList<>();
         Mat hierarchy = new Mat();
         Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);
    	 return contours;
    }
	
	private List<MatOfPoint> obtainContoursNotAtBorder (List<MatOfPoint> contours, Mat src) {
		List<MatOfPoint> newContours = new ArrayList<>();
        for (int contourID = 0; contourID < contours.size(); contourID++)
        {
            boolean atBorder = checkContourAtBorder(contours.get(contourID), src);
            Double area = Imgproc.contourArea(contours.get(contourID));
//            System.out.println(area);
            double contourArea = Imgproc.contourArea(contours.get(contourID));
            if (atBorder == true || area < (src.width()*src.height())/50) {
//            	contours.remove(contourID);
            } else {
            	newContours.add(contours.get(contourID));
            }
        }
    	return newContours;
    }
	
    private boolean checkContourAtBorder(MatOfPoint contour, Mat src) {
    	boolean atBorder = false;
        Point[] points = contour.toArray();
        BitSet bs = new BitSet(4);
        for (Point p : points) {
        	if (p.x == 0) {
        		bs.set(0);
        	}
        	if (p.x == src.width()-1) {
        		bs.set(1);
        	}
        	if (p.y == 0) {
        		bs.set(2);
        	}
        	if (p.y == src.height()-1) {
        		bs.set(3);
        	}
        }
        int hitCount = bs.cardinality();
        if (hitCount > 2) {
        	atBorder = true;
        }
    	return atBorder;
    }
	
	private List<Mat> channelSplitterAndThresholder (Mat src, int kSize) {
		Mat ch1 = new Mat();
		Core.extractChannel(src.clone(), ch1, 0);
		Mat ch2 = new Mat();
		Core.extractChannel(src.clone(), ch2, 1);
		Mat ch3 = new Mat();
		Core.extractChannel(src.clone(), ch3, 2);
		
		List<Mat> channels = Arrays.asList(thresholder(ch1, kSize), thresholder(ch2, kSize), thresholder(ch3, kSize));
		return channels;
	}
	
	private Mat thresholder (Mat src, int kSize) {
		Mat blur = new Mat();
		Imgproc.medianBlur(src.clone(), blur, kSize);
		Mat dst = new Mat();
		Imgproc.threshold(blur, dst, 0, 255, Imgproc.THRESH_OTSU);
		return dst;
	}
	
	public static List<Mat> cluster(Mat cutout, int k) {
		Mat samples = cutout.reshape(1, cutout.cols() * cutout.rows());
		Mat samples32f = new Mat();
		samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);
		
		Mat labels = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
		Mat centers = new Mat();
		Core.kmeans(samples32f, k, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);		
		return showClusters(cutout, labels, centers);
	}

	private static List<Mat> showClusters (Mat cutout, Mat labels, Mat centers) {
		centers.convertTo(centers, CvType.CV_8UC1, 255.0);
		centers.reshape(3);
		
		List<Mat> clusters = new ArrayList<Mat>();
		for(int i = 0; i < centers.rows(); i++) {
			clusters.add(Mat.zeros(cutout.size(), cutout.type()));
		}
		
		Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
		for(int i = 0; i < centers.rows(); i++) counts.put(i, 0);
		
		int rows = 0;
		for(int y = 0; y < cutout.rows(); y++) {
			for(int x = 0; x < cutout.cols(); x++) {
				int label = (int)labels.get(rows, 0)[0];
				int r = (int)centers.get(label, 2)[0];
				int g = (int)centers.get(label, 1)[0];
				int b = (int)centers.get(label, 0)[0];
				counts.put(label, counts.get(label) + 1);
				clusters.get(label).put(y, x, b, g, r);
				rows++;
			}
		}
//		System.out.println(counts);
		return clusters;
	}
	
	
}
