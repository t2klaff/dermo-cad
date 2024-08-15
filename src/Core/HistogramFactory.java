package Core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class HistogramFactory {
	public double rMeanLesion;
	public double gMeanLesion;
	public double bMeanLesion;
	public double rMedianLesion;
	public double gMedianLesion;
	public double bMedianLesion;
	public double rStdLesion;
	public double gStdLesion;
	public double bStdLesion;
	public double rMeanSkin;
	public double gMeanSkin;
	public double bMeanSkin;
	public double rMedianSkin;
	public double gMedianSkin;
	public double bMedianSkin;
	public double rStdSkin;
	public double gStdSkin;
	public double bStdSkin;
	
	public double lMeanLesion;
	public double aMeanLesion;
	public double bbMeanLesion;
	public double lMedianLesion;
	public double aMedianLesion;
	public double bbMedianLesion;
	public double lStdLesion;
	public double aStdLesion;
	public double bbStdLesion;
	public double lMeanSkin;
	public double aMeanSkin;
	public double bbMeanSkin;
	public double lMedianSkin;
	public double aMedianSkin;
	public double bbMedianSkin;
	public double lStdSkin;
	public double aStdSkin;
	public double bbStdSkin;
	
	public HistogramFactory(Mat src, Mat Segmentation)
	{

		Mat SegmentationNot = new Mat();
		Core.bitwise_not(Segmentation.clone(), SegmentationNot);
//		Imgproc.cvtColor(SegmentationNot, SegmentationNot, Imgproc.COLOR_BGR2GRAY);
		
		List<Double> bgrHistogramLesionFeatures = getHistogram(src.clone(), Segmentation.clone());
		bMeanLesion = bgrHistogramLesionFeatures.get(0);
		bMedianLesion = bgrHistogramLesionFeatures.get(1);
		bStdLesion = bgrHistogramLesionFeatures.get(2);
		gMeanLesion = bgrHistogramLesionFeatures.get(3);
		gMedianLesion = bgrHistogramLesionFeatures.get(4);
		gStdLesion = bgrHistogramLesionFeatures.get(5);
		rMeanLesion = bgrHistogramLesionFeatures.get(6);
		rMedianLesion = bgrHistogramLesionFeatures.get(7);
		rStdLesion = bgrHistogramLesionFeatures.get(8);
		List<Double> bgrHistogramSkinFeatures = getHistogram(src.clone(), SegmentationNot.clone());
		bMeanSkin = bgrHistogramSkinFeatures.get(0);
		bMedianSkin = bgrHistogramSkinFeatures.get(1);
		bStdSkin = bgrHistogramSkinFeatures.get(2);
		gMeanSkin = bgrHistogramSkinFeatures.get(3);
		gMedianSkin = bgrHistogramSkinFeatures.get(4);
		gStdSkin = bgrHistogramSkinFeatures.get(5);
		rMeanSkin = bgrHistogramSkinFeatures.get(6);
		rMedianSkin = bgrHistogramSkinFeatures.get(7);
		rStdSkin = bgrHistogramSkinFeatures.get(8);
		
		Mat lab = new Mat();
		Imgproc.cvtColor(src.clone(), lab, Imgproc.COLOR_BGR2Lab);
		
		List<Double> labHistogramLesionFeatures = getHistogram(lab.clone(), Segmentation.clone());
		lMeanLesion = labHistogramLesionFeatures.get(0);
		lMedianLesion = labHistogramLesionFeatures.get(1);
		lStdLesion = labHistogramLesionFeatures.get(2);
		aMeanLesion = labHistogramLesionFeatures.get(3);
		aMedianLesion = labHistogramLesionFeatures.get(4);
		aStdLesion = labHistogramLesionFeatures.get(5);
		bbMeanLesion = labHistogramLesionFeatures.get(6);
		bbMedianLesion = labHistogramLesionFeatures.get(7);
		bbStdLesion = labHistogramLesionFeatures.get(8);
		List<Double> labHistogramSkinFeatures = getHistogram(lab.clone(), SegmentationNot.clone());
		lMeanSkin = labHistogramSkinFeatures.get(0);
		lMedianSkin = labHistogramSkinFeatures.get(1);
		lStdSkin = labHistogramSkinFeatures.get(2);
		aMeanSkin = labHistogramSkinFeatures.get(3);
		aMedianSkin = labHistogramSkinFeatures.get(4);
		aStdSkin = labHistogramSkinFeatures.get(5);
		bbMeanSkin = labHistogramSkinFeatures.get(6);
		bbMedianSkin = labHistogramSkinFeatures.get(7);
		bbStdSkin = labHistogramSkinFeatures.get(8);
	}
	
	private List<Double> getHistogram (Mat src, Mat Segmentation) {
    	
		// split the image into channels
        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(src, bgrPlanes);

        // set number of bins
        int histSize = 256;

        // set the range
        float[] range = {0, 256};
        MatOfFloat histRange = new MatOfFloat(range);
        boolean accumulate = false;

        // get histogram
        Mat bHist = new Mat(), gHist = new Mat(), rHist = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), Segmentation.clone(), bHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(1), Segmentation.clone(), gHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(2), Segmentation.clone(), rHist, new MatOfInt(histSize), histRange, accumulate);

        // create histogram matrix
        int histW = 600, histH = 450;
        int binW = (int) Math.round((double) histW / histSize);
        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );

        // normalise histogram
        Core.normalize(bHist, bHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(gHist, gHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(rHist, rHist, 0, histImage.rows(), Core.NORM_MINMAX);
        float[] bHistData = new float[(int) (bHist.total() * bHist.channels())];
        bHist.get(0, 0, bHistData);
        float[] gHistData = new float[(int) (gHist.total() * gHist.channels())];
        gHist.get(0, 0, gHistData);
        float[] rHistData = new float[(int) (rHist.total() * rHist.channels())];
        rHist.get(0, 0, rHistData);

        // draw histogram to matrix
        for( int i = 1; i < histSize; i++ ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
        }

//        Mat masked = obtainMasked (src, Segmentation);
        
        List<Double> bHistStats = calculateStats(bHistData);
        List<Double> gHistStats = calculateStats(gHistData);
        List<Double> rHistStats = calculateStats(rHistData);
        List<Double> results = Stream.of(bHistStats, gHistStats, rHistStats).flatMap(Collection::stream).collect(Collectors.toList());
    	
    	return results;
    }
	
	public Mat getBGRHistogram (Mat src, Mat Segmentation) {
    	
		// split the image into channels
        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(src, bgrPlanes);

        // set number of bins
        int histSize = 256;

        // set the range
        float[] range = {0, 256};
        MatOfFloat histRange = new MatOfFloat(range);
        boolean accumulate = false;

        // get histogram
        Mat bHist = new Mat(), gHist = new Mat(), rHist = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), Segmentation.clone(), bHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(1), Segmentation.clone(), gHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(2), Segmentation.clone(), rHist, new MatOfInt(histSize), histRange, accumulate);

        // create histogram matrix
        int histW = 600, histH = 450;
        int binW = (int) Math.round((double) histW / histSize);
        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );

        // normalise histogram
        Core.normalize(bHist, bHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(gHist, gHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(rHist, rHist, 0, histImage.rows(), Core.NORM_MINMAX);
        float[] bHistData = new float[(int) (bHist.total() * bHist.channels())];
        bHist.get(0, 0, bHistData);
        float[] gHistData = new float[(int) (gHist.total() * gHist.channels())];
        gHist.get(0, 0, gHistData);
        float[] rHistData = new float[(int) (rHist.total() * rHist.channels())];
        rHist.get(0, 0, rHistData);

        // draw histogram to matrix
        for( int i = 1; i < histSize; i++ ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
        }

//        Mat masked = obtainMasked (src, Segmentation);
    	
    	return histImage.clone();
    }
    
    private List<Double> calculateStats(float array[])
    {
    	List<Double> stats = new ArrayList<>();
        double sum = 0.0;
        double mean = 0.0;
        double median = 0.0;
        double standardDeviation = 0.0;
        int length = array.length;

        for(float f : array) {
            sum += f;
        }

        mean = sum/length;
        stats.add(mean);
        
        median = Double.valueOf(String.valueOf((array[array.length/2] + array[(array.length-1)/2]) / 2));
        stats.add(median);
        
        for(float f: array) {
            standardDeviation += Math.pow(f - mean, 2);
        }
        
        standardDeviation = Math.sqrt(standardDeviation/length);
        stats.add(standardDeviation);

        return stats;
    }
	
}
