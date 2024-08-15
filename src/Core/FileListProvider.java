package Core;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FileListProvider {
	private static String[] validExtensions = {".png", ".jpg"};
	private static Integer _maxFiles = 100000;
	
	public List<String> getFiles(String folderName) {

    	var files = new File(folderName).listFiles();
		List<String> fileNames = new ArrayList<>();
		
		for (var f : files) {
			String filePath = f.getAbsolutePath();
			String extension = getExtension(filePath);
			if (Arrays.asList(validExtensions).contains(extension))
			{
				fileNames.add(filePath);
				if (fileNames.size() == _maxFiles) {
					break;
				}
			}
		}	
		return fileNames;
	}

	private String getExtension(String filePath) {
		Path path = Paths.get(filePath);
		String fileName = path.getFileName().toString();
		int index = fileName.lastIndexOf('.');
		if (index < 0)
			return "";
		return fileName.substring(index, fileName.length());
	}
}