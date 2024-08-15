package Core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class SkinTypeMapper {

	private String _hamPath = "E:\\Final Year Project Dataset\\WORKING\\Training_HAM10000";
	private String _bcnPath = "E:\\Final Year Project Dataset\\WORKING\\Training_BCN20000";
	private String _mskPath = "E:\\Final Year Project Dataset\\WORKING\\Training_MSK";
	
	public void mapBySkinType() throws IOException {
		
		List<String> hamFileList = new FileListProvider().getFiles(_hamPath);
		List<String> bcnFileList = new FileListProvider().getFiles(_bcnPath);
		List<String> mskFileList = new FileListProvider().getFiles(_mskPath);
		
//		System.out.println(hamFileList.size());
//		System.out.println(bcnFileList.size());
//		System.out.println(mskFileList.size());
		
		Map<String, Double> hamSkinTypeMap = new HashMap<>();
		Map<String, String> hamSkinGroupMap = new HashMap<>();
		
		int hamCount = 1;
		for (var f : hamFileList)
		{
//			if (count > 0) {
//				break;
//			}
			
			String fileName = getFileName(f);
			ImagesFactory imagesFactory = new ImagesFactory(f, false);
			
			Mat src = imagesFactory.GetMatrix(ImagesFactory.BorderRemovedDs);
			Mat seg = imagesFactory.GetMatrix(ImagesFactory.SegmentationDs);
			
//			Imgproc.cvtColor(seg, seg, Imgproc.COLOR_BGR2GRAY);
			
    		Mat segNot = new Mat();
    		Core.bitwise_not(seg, segNot);
    		
			Imgproc.cvtColor(segNot, segNot, Imgproc.COLOR_GRAY2BGR);
    		Mat masked = new Mat();
    		Core.bitwise_and(src, segNot, masked);
    		
//    		HighGui.imshow("masked", masked);
//    		HighGui.waitKey();
    		
    		Mat lab = new Mat();
    		Imgproc.cvtColor(masked, lab, Imgproc.COLOR_BGR2Lab);
    		
    		Mat L = new Mat();
    		Core.extractChannel(lab, L, 0);
    		
    		Mat b = new Mat();
    		Core.extractChannel(lab, b, 2);
    		
    		List<Double> LVals = new ArrayList<>();
    		List<Double> bVals = new ArrayList<>();
    		
    		for (int p = 0; p < lab.cols(); p ++) {
	        	for (int q = 0; q < lab.rows(); q ++) {
	        		double labVal = lab.get(q,p)[0];
	        		if (labVal != 0) {
	        			LVals.add(L.get(q,p)[0]);
	        			bVals.add(b.get(q,p)[0]);
	        		}
	        	}
	        }
    		
        	Double LAverage = LVals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        	Double bAverage = bVals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        	double ita = Math.atan((LAverage - 50) / bAverage) * (180/Math.PI);
        	hamSkinTypeMap.put(fileName, ita);
//        	System.out.println("ita: " + ita);
        	
//        	Path ogPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\Training_HAM10000\\" + fileName + ".jpg");
//        	Path copyPath = null;
        	String group = null;
        	
        	if (ita > 55) {
        		group = "I";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\I\\" + fileName + ".jpg");
        	} else if (ita > 41) {
        		group = "II";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\II\\" + fileName + ".jpg");
        	} else if (ita > 28) {
        		group = "III";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\III\\" + fileName + ".jpg");
        	} else if (ita > 10) {
        		group = "IV";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\IV\\" + fileName + ".jpg");
        	} else if (ita > -30) {
        		group = "V";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\V\\" + fileName + ".jpg");
        	} else { //ita < -30
        		group = "VI";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\VI\\" + fileName + ".jpg");
        	}
        	hamSkinGroupMap.put(fileName, group);
        	System.out.println(hamCount + "/" + hamFileList.size() + " - " + fileName + " - group: " + group);
//        	Files.copy(ogPath, copyPath);
        	hamCount++;
		}
		
		
		try (BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\Final Year Project Dataset\\WORKING\\hamSkinTypeMap.txt"))) {
            for (Map.Entry<String, Double> entry : hamSkinTypeMap.entrySet()) {
                bf.write(entry.getKey() + "," + hamSkinGroupMap.get(entry.getKey()) + "," + entry.getValue());
                bf.newLine();
            }
            bf.flush();
		}
		
		Map<String, Double> bcnSkinTypeMap = new HashMap<>();
		Map<String, String> bcnSkinGroupMap = new HashMap<>();
		int bcnCount = 1;
		for (var f : bcnFileList)
		{
//			if (count > 2) {
//				break;
//			}
			
			String fileName = getFileName(f);
			ImagesFactory imagesFactory = new ImagesFactory(f, false);
			
			Mat src = imagesFactory.GetMatrix(ImagesFactory.BorderRemovedDs);
			Mat seg = imagesFactory.GetMatrix(ImagesFactory.SegmentationDs);
			
//    		HighGui.imshow("src", src);
//    		HighGui.imshow("seg", seg);
			
//			Imgproc.cvtColor(seg, seg, Imgproc.COLOR_BGR2GRAY);
    		Mat segNot = new Mat();
    		Core.bitwise_not(seg, segNot);
    		
			Imgproc.cvtColor(segNot, segNot, Imgproc.COLOR_GRAY2BGR);
    		Mat masked = new Mat();
    		Core.bitwise_and(src, segNot, masked);
    		
//    		HighGui.imshow("masked", masked);
//    		HighGui.waitKey();
    		
    		Mat lab = new Mat();
    		Imgproc.cvtColor(masked, lab, Imgproc.COLOR_BGR2Lab);
    		
    		Mat L = new Mat();
    		Core.extractChannel(lab, L, 0);
    		
    		Mat b = new Mat();
    		Core.extractChannel(lab, b, 2);
    		
    		List<Double> LVals = new ArrayList<>();
    		List<Double> bVals = new ArrayList<>();
    		
    		for (int p = 0; p < lab.cols(); p ++) {
	        	for (int q = 0; q < lab.rows(); q ++) {
	        		double labVal = lab.get(q,p)[0];
	        		if (labVal != 0) {
	        			LVals.add(L.get(q,p)[0]);
	        			bVals.add(b.get(q,p)[0]);
	        		}
	        	}
	        }
    		
        	Double LAverage = LVals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        	Double bAverage = bVals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        	double ita = Math.atan((LAverage - 50) / bAverage) * (180/Math.PI);
        	bcnSkinTypeMap.put(fileName, ita);
//        	System.out.println("ita: " + ita);
        	
//        	Path ogPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\Training_HAM10000\\" + fileName + ".jpg");
//        	Path copyPath = null;
        	String group = null;
        	
        	if (ita > 55) {
        		group = "I";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\I\\" + fileName + ".jpg");
        	} else if (ita > 41) {
        		group = "II";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\II\\" + fileName + ".jpg");
        	} else if (ita > 28) {
        		group = "III";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\III\\" + fileName + ".jpg");
        	} else if (ita > 10) {
        		group = "IV";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\IV\\" + fileName + ".jpg");
        	} else if (ita > -30) {
        		group = "V";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\V\\" + fileName + ".jpg");
        	} else { //ita < -30
        		group = "VI";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\VI\\" + fileName + ".jpg");
        	}
        	bcnSkinGroupMap.put(fileName, group);
        	System.out.println(bcnCount + "/" + bcnFileList.size() + " - " + fileName + " - group: " + group);
//        	Files.copy(ogPath, copyPath);
        	bcnCount++;
		}
		
		try (BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\Final Year Project Dataset\\WORKING\\bcnSkinTypeMap.txt"))) {
            for (Map.Entry<String, Double> entry : bcnSkinTypeMap.entrySet()) {
                bf.write(entry.getKey() + "," + bcnSkinGroupMap.get(entry.getKey()) + "," + entry.getValue());
                bf.newLine();
            }
            bf.flush();
        }
		
		Map<String, Double> mskSkinTypeMap = new HashMap<>();
		Map<String, String> mskSkinGroupMap = new HashMap<>();
		int mskCount = 1;
		for (var f : mskFileList)
		{
			String fileName = getFileName(f);
			ImagesFactory imagesFactory = new ImagesFactory(f, false);
			
			Mat src = imagesFactory.GetMatrix(ImagesFactory.BorderRemovedDs);
			Mat seg = imagesFactory.GetMatrix(ImagesFactory.SegmentationDs);
			
//			Imgproc.cvtColor(seg, seg, Imgproc.COLOR_BGR2GRAY);
    		Mat segNot = new Mat();
    		Core.bitwise_not(seg, segNot);
    		
			Imgproc.cvtColor(segNot, segNot, Imgproc.COLOR_GRAY2BGR);
    		Mat masked = new Mat();
    		Core.bitwise_and(src, segNot, masked);
    		
    		Mat lab = new Mat();
    		Imgproc.cvtColor(masked, lab, Imgproc.COLOR_BGR2Lab);
    		
    		Mat L = new Mat();
    		Core.extractChannel(lab, L, 0);
    		
    		Mat b = new Mat();
    		Core.extractChannel(lab, b, 2);
    		
    		List<Double> LVals = new ArrayList<>();
    		List<Double> bVals = new ArrayList<>();
    		
    		for (int p = 0; p < lab.cols(); p ++) {
	        	for (int q = 0; q < lab.rows(); q ++) {
	        		double labVal = lab.get(q,p)[0];
	        		if (labVal != 0) {
	        			LVals.add(L.get(q,p)[0]);
	        			bVals.add(b.get(q,p)[0]);
	        		}
	        	}
	        }
    		
        	Double LAverage = LVals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        	Double bAverage = bVals.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
        	double ita = Math.atan((LAverage - 50) / bAverage) * (180/Math.PI);
        	mskSkinTypeMap.put(fileName, ita);
//        	System.out.println("ita: " + ita);
        	
//        	Path ogPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\Training_HAM10000\\" + fileName + ".jpg");
//        	Path copyPath = null;
        	String group = null;
        	
        	if (ita > 55) {
        		group = "I";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\I\\" + fileName + ".jpg");
        	} else if (ita > 41) {
        		group = "II";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\II\\" + fileName + ".jpg");
        	} else if (ita > 28) {
        		group = "III";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\III\\" + fileName + ".jpg");
        	} else if (ita > 10) {
        		group = "IV";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\IV\\" + fileName + ".jpg");
        	} else if (ita > -30) {
        		group = "V";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\V\\" + fileName + ".jpg");
        	} else { //ita < -30
        		group = "VI";
//        		copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\HAM10000_by_skintype\\VI\\" + fileName + ".jpg");
        	}
        	mskSkinGroupMap.put(fileName, group);
        	System.out.println(mskCount + "/" + mskFileList.size() + " - " + fileName + " - group: " + group);
//        	Files.copy(ogPath, copyPath);
        	mskCount++;
		}
		
		try (BufferedWriter bf = new BufferedWriter(new FileWriter("E:\\Final Year Project Dataset\\WORKING\\mskSkinTypeMap.txt"))) {
            for (Map.Entry<String, Double> entry : mskSkinTypeMap.entrySet()) {
                bf.write(entry.getKey() + "," + mskSkinGroupMap.get(entry.getKey()) + "," + entry.getValue());
                bf.newLine();
            }
            bf.flush();
        }
		
		
	}
	
	private String getFileName(String filePath) {
		Path path = Paths.get(filePath);
		String fileName = path.getFileName().toString();
		return fileName.substring(0, fileName.lastIndexOf('.'));
	}
	
}
