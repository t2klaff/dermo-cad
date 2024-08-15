package Features;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class FractalDimensionFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getFractalDimension(imagesFactory, imagesFactory.GetMatrix(ImagesFactory.SegmentationV3), imagesFactory.GetMatrix(ImagesFactory.SegmentationOutlineBorder));
	}
	
	private double getFractalDimension (ImagesFactory imagesFactory, Mat Segmentation, Mat SegmentationOutlineBorder) {
		
        int addHeight = ((int)SegmentationOutlineBorder.height() - Segmentation.rows())/2;
        int addWidth = ((int)SegmentationOutlineBorder.width() - Segmentation.cols())/2;
        Rect ogROI = new Rect(addWidth, addHeight, Segmentation.cols(), Segmentation.rows());
//    	System.out.println(SegmentationOutlineBorder.size());
    	
        List<Integer> boxSizes = Arrays.asList(1, 2, 4, 8, 16, 32, 64, 128);
        List<Integer> boxCounts = new ArrayList<>();
        List<Double> logr = new ArrayList<>(boxSizes.size());
        List<Double> logNr = new ArrayList<>(boxSizes.size());
        List<Double> product = new ArrayList<>(boxSizes.size());
        List<Double> logr2 = new ArrayList<>(boxSizes.size());
        List<Double> logNr2 = new ArrayList<>(boxSizes.size());
        List<Mat> imgs = new ArrayList<>();
        
        // for all the box sizes
        for (int p = 0; p < boxSizes.size(); p++) {
        	int boxSize = boxSizes.get(p);
//        	System.out.println("Calculating N(r) with r = " + boxSize);
        	Mat result = SegmentationOutlineBorder.clone();
        	Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR);
            int width = (int)SegmentationOutlineBorder.size().width;
            int height = (int)SegmentationOutlineBorder.size().height;
            int boxCount = 0;
            // count how many boxes cover any boundary point
	        for (int i = 0; i < width; i += boxSize) {
	        	for (int j = 0; j < height; j += boxSize) {
	        		Rect rect = new Rect(i, j, boxSize, boxSize);
	        		Mat roi = new Mat(SegmentationOutlineBorder, rect);
	        		Scalar mean = Core.mean(roi);
	        		double sMean = mean.val[0];
	        		if (sMean > 0) {
	        			boxCount++;
	        			Imgproc.rectangle(result, rect, new Scalar(255,255,0), 1);
	              }
	        	}
	        }
	        logr.add(p, -1*(Math.log(boxSizes.get(p))));
	        logNr.add(p, Math.log(boxCount));
	        product.add(p, (-1*(Math.log(boxSizes.get(p))))*Math.log(boxCount));
	        logr2.add(p, (-1*(Math.log(boxSizes.get(p))))*(-1*(Math.log(boxSizes.get(p)))));
	        logNr2.add(p, Math.log(boxCount)*Math.log(boxCount));
	        boxCounts.add(boxCount);
            imgs.add(result);
        } 
        
        // drawing images
        List<Mat> imgs2 = new ArrayList<>();
    	for (int m = 0; m < imgs.size(); m++) {
            Mat xRoi = new Mat(imgs.get(m), ogROI);
            Imgproc.putText(xRoi, "r: " + String.valueOf(boxSizes.get(m)), new Point(10, 420), Imgproc.FONT_HERSHEY_COMPLEX_SMALL, 1, new Scalar(0,0,255), 1);
            Imgproc.putText(xRoi, "N(r): " + String.valueOf(boxCounts.get(m)), new Point(10, 440), Imgproc.FONT_HERSHEY_COMPLEX_SMALL, 1, new Scalar(0,0,255), 1);
            imgs2.add(xRoi);
            imagesFactory.AddMatrix("fractalDimension" + String.valueOf(boxSizes.get(m)), xRoi);
    	}
    	List<Mat> list1 = imgs2.subList(0, imgs2.size()/2);
        List<Mat> list2 = imgs2.subList(imgs2.size()/2, imgs2.size());
        Mat list1Mat = new Mat();
        Mat list2Mat = new Mat();
        Core.hconcat(list1, list1Mat);
        Core.hconcat(list2, list2Mat);
        List<Mat> listMats = new ArrayList<>();
        listMats.add(list1Mat);
        listMats.add(list2Mat);
        Mat dst = new Mat();
        Core.vconcat(listMats, dst);
        imagesFactory.AddMatrix("fractalDimension", dst.clone());
//        HighGui.imshow("box counting", dst);
        
        double logrSum = logr.stream().mapToDouble(Double::doubleValue).sum();
        double logNrSum = logNr.stream().mapToDouble(Double::doubleValue).sum();
        double productSum = product.stream().mapToDouble(Double::doubleValue).sum();
        double logr2Sum = logr2.stream().mapToDouble(Double::doubleValue).sum();
//        double logNr2Sum = logNr.stream().mapToDouble(Double::doubleValue).sum();
        double df = ((boxSizes.size()*productSum)-(logrSum*logNrSum)) / ((boxSizes.size()*logr2Sum)-(logrSum*logrSum));
        
    	return df;
    }
}
