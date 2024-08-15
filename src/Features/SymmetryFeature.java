package Features;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class SymmetryFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
        RotatedRect minBoundingRectBorder = Imgproc.minAreaRect(contoursFactory.Contour2fBorder);
		return getSymmetry(imagesFactory, minBoundingRectBorder);
	}
	
	private Double getSymmetry (ImagesFactory imagesFactory, RotatedRect minBoundingRect) {
		
		Mat lesionMask = imagesFactory.GetMatrix(ImagesFactory.SegmentationBorder);
		
        Mat rotated = new Mat();
        Mat cropped = new Mat();
        
        // get RotatedRect
        double angle = minBoundingRect.angle;
        Point center = minBoundingRect.center;
        Size rect_size = minBoundingRect.size;
        if (minBoundingRect.angle < -45.) {
        	   angle += 90.0;
        	   Double widthTmp = rect_size.width;
        	   Double heightTmp = rect_size.height;
        	   rect_size.width = heightTmp;
        	   rect_size.height = widthTmp;
        }
        
        // draw the workings
        Mat drawing = lesionMask.clone();
        Imgproc.cvtColor(drawing, drawing, Imgproc.COLOR_GRAY2BGR);
        Point points[] = new Point[4];
        minBoundingRect.points(points);
        for(int i=0; i<4; ++i){
            Imgproc.line(drawing, points[i], points[(i+1)%4], new Scalar(255,0,255), 2);
        }
//        System.out.println("Detected angle: " + angle);

        // get rotation matrix
        Mat rot_mat = Imgproc.getRotationMatrix2D(center, angle, 1);
        Imgproc.warpAffine(lesionMask, rotated, rot_mat, lesionMask.size(), Imgproc.INTER_NEAREST);
        Imgproc.getRectSubPix(rotated, rect_size, center, cropped);
        Imgproc.threshold(cropped, cropped, 0, 255, Imgproc.THRESH_BINARY);
        
        // flipping on primary axis
        Mat flipped = cropped.clone();
        int flipCode;
        if (flipped.width() > flipped.height()){
        	flipCode = 0;
        } else {
        	flipCode = 1;
        }
    	Core.flip(flipped, flipped, flipCode);
//        System.out.println("flipCode: " + flipCode);
    	
		Mat aMat = new Mat();
		Core.bitwise_or(cropped, flipped, aMat);
		Mat fsMat = new Mat();
		Core.bitwise_xor(cropped, aMat, fsMat);
        double count_white_a= Core.countNonZero(aMat);
        double count_white_fs= Core.countNonZero(fsMat);
        double symmetry = 1 - (count_white_fs/count_white_a);
        
        imagesFactory.AddMatrix("minBoundingRect", drawing.clone());
        imagesFactory.AddMatrix("symmetryCropped", cropped.clone());
        imagesFactory.AddMatrix("symmetryRotated", rotated.clone());
        imagesFactory.AddMatrix("symmetryFlipped", flipped.clone());
        imagesFactory.AddMatrix("symmetryAMat", aMat.clone());
        imagesFactory.AddMatrix("symmetryFsMat", fsMat.clone());
        
    	return symmetry;
    }
}
