package Core;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DiagnosisMapper {
	
	private static Map<String, String> _diagnosisMap = null;
	
	public DiagnosisMapper(String metadataPath) throws IOException {
		// only get diagnoses if needed (i.e only for training folder not test folder)
		if (_diagnosisMap == null && metadataPath != null) {
			_diagnosisMap = obtainAllDiagnoses(metadataPath);
		}
	}
	
	public String getDiagnosis(String filePath) {
		if (_diagnosisMap == null)
		{
			// unknown class for Weka prediction
			return "?";
		}
		String fileName = getFilenameWithoutExtension(filePath);
		return _diagnosisMap.get(fileName);
	}
	
	private Map<String, String> obtainAllDiagnoses(String metadataPath) throws IOException {
        FileReader fileReader = new FileReader(metadataPath);
        System.out.println("ground truth: [" + metadataPath + "]");
	        
        Map<String, String> diagnosesMap = new HashMap<String, String>();
        List<String> firstLine = new ArrayList<String>();
        
        int lineCount = 0;
//        int benignCount = 0;
//        int malignantCount = 0;
        
        // sorting the metadata into 2 classes malignant / benign
        try (BufferedReader bufferedReader = new BufferedReader(fileReader)) {
            String line;
            while((line = bufferedReader.readLine()) != null) {
        		List<String> lineArray = Arrays.asList(line.split(","));
            	String diag = null;
            	if (lineCount == 0) {
            		firstLine = lineArray;
            	} else {
            		String imgID = lineArray.get(0);
//                    System.out.println(imgID);
                	for (int i=1; i<lineArray.size()-1; i++) {
                		if (Double.parseDouble(lineArray.get(i)) == 1.0) {
                			diag = firstLine.get(i);
                		}
                	}
//                    System.out.println(diag);
                	if (diag.equals("MEL") || diag.equals("VASC") || diag.equals("SCC")) {
//                		malignantCount++;
                		diagnosesMap.put(imgID, "MALIGNANT");
                	} else if (diag.equals("DF") || diag.equals("AK") || diag.equals("NV") || diag.equals("BCC") || diag.equals("BKL")){
//                		benignCount++;
                		diagnosesMap.put(imgID, "BENIGN");
                	} else {
                		diagnosesMap.put(imgID, "UNK");
                		System.out.println("error: unknown diagnosis");
                	}
                	
            	}
                lineCount++;
            }
        }
        
//        System.out.println("benignCount: " + benignCount);
//        System.out.println("malignantCount: " + malignantCount);
    	return diagnosesMap;
	}

	private String getFilenameWithoutExtension(String filePath) {
		Path path = Paths.get(filePath);
		String fileName = path.getFileName().toString();
		return fileName.substring(0, fileName.lastIndexOf('.'));
	}
}
