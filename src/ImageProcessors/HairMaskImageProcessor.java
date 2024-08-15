package ImageProcessors;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

public class HairMaskImageProcessor {
	public Mat getMatrix(Mat src)
	{
		Mat gray = new Mat();
      
      Core.extractChannel(src, gray, 2);
      
      // blur and denoise grayscale image
      Mat blur = new Mat();
      Imgproc.medianBlur(gray, blur, 3);
//      Imgproc.GaussianBlur(gray, blur, new Size(3,3), 0);
      Photo.fastNlMeansDenoising(blur, blur, 3, 7, 21);
      
      // laplacian filter and denoise
      Mat lap = new Mat();
      Imgproc.Laplacian(blur, lap, CvType.CV_16S, 3, 1, 0, Core.BORDER_DEFAULT );
      Core.convertScaleAbs( lap, lap );
      Photo.fastNlMeansDenoising(lap, lap, 5, 7, 21);
      
      // subtract laplace from grayscale
      Mat sub = new Mat();
      Core.subtract(gray, lap, sub, new Mat(), -1);
      
      // canny edge detection
      Mat canny = new Mat();
      Imgproc.Canny(sub, canny, 50, 150, 3);
      
      // refinement morphology
      Mat morph = new Mat();
      int kernelSize1 = 5;
      Mat element1 = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_ELLIPSE, new Size(2 * kernelSize1 + 1, 2 * kernelSize1 + 1), new Point(kernelSize1, kernelSize1));
      Imgproc.morphologyEx(canny, morph, Imgproc.MORPH_CLOSE, element1);
      
      // hough lines detection
      Mat linesP = new Mat();
      Imgproc.HoughLinesP(morph, linesP, 1, Math.PI/180, 20, 15, 10);
      
      Mat zeros = Mat.zeros(src.size(), src.type());
      
      // draw lines detected by hough lines
      for (int x = 0; x < linesP.rows(); x++) {
          double[] l = linesP.get(x, 0);
          Imgproc.line(zeros, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 255, 255), 2, Imgproc.LINE_AA, 0);
      }
      
      // final refinement morphology
      Mat morph2 = new Mat();
      int kernelSize3 = 2;
      Mat element3 = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_ELLIPSE, new Size(2 * kernelSize3 + 1, 2 * kernelSize3 + 1), new Point(kernelSize3, kernelSize3));
      Imgproc.morphologyEx(zeros, morph2, Imgproc.MORPH_CLOSE, element3);
       
      Mat hairMask = morph2.clone();
//	  HighGui.imshow("hairMask", hairMask);
//	  HighGui.waitKey();
      
      Imgproc.cvtColor(hairMask, hairMask, Imgproc.COLOR_BGR2GRAY);
      
  	return hairMask;
	}
}
