package Core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class SkinTypeSorter {

	private String _hamPath = "E:\\Final Year Project Dataset\\WORKING\\Training_HAM10000";
	private String _bcnPath = "E:\\Final Year Project Dataset\\WORKING\\Training_BCN20000";
	private String _mskPath = "E:\\Final Year Project Dataset\\WORKING\\Training_MSK";
	
	public void sortBySkinType(String skinTypeMap, String inputFolderName, String outputFolderName) throws IOException {
		
//        FileReader fileReader = new FileReader("E:\\Final Year Project Dataset\\WORKING\\mskSkinTypeMap.txt");
        FileReader fileReader = new FileReader("E:\\Final Year Project Dataset\\WORKING\\" + skinTypeMap);
        
        int lineCount = 0;
        try (BufferedReader bufferedReader = new BufferedReader(fileReader)) {
            String line;
            while((line = bufferedReader.readLine()) != null) {
        		List<String> lineArray = Arrays.asList(line.split(","));
        		String name = lineArray.get(0);
//        		String ogFile = "E:\\Final Year Project Dataset\\WORKING\\Training_MSK\\" + name + ".jpg";
        		String ogFile = "E:\\Final Year Project Dataset\\WORKING\\" + inputFolderName + "\\" + name + ".jpg";
        		Path ogPath = Paths.get(ogFile);
        		String group = lineArray.get(1);
//        		Path copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\MSK_by_skintype\\" + group + "\\" + name + ".jpg");
        		Path copyPath = Paths.get("E:\\Final Year Project Dataset\\WORKING\\" + outputFolderName + "\\" + group + "\\" + name + ".jpg");
        		System.out.println("copying " + name + " to " + group);
            	Files.copy(ogPath, copyPath);
                lineCount++;
            }
        }
	
	}
	
	private String getFileName(String filePath) {
		Path path = Paths.get(filePath);
		String fileName = path.getFileName().toString();
		return fileName.substring(0, fileName.lastIndexOf('.'));
	}
	
}
