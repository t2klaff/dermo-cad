import Core.Coordinator;

public class Startup {
	private static String _defaultTrainingFolder = ".\\data\\Training";
	private static String _defaultTestFolder = ".\\data\\Test";
	private static String _defaultOutputFolder = ".\\data\\Output";
	
	// *
	// dependencies: opencv-455.jar (OpenCV 4.5.5)
	//				 weka.jar (Weka 3.8.6)
	//				 SMOTE.jar
	// *
	
	public static void main(String[] args) throws Exception {
		if (args.length == 0)
		{
			new Coordinator().run(_defaultTrainingFolder, _defaultTestFolder, _defaultOutputFolder, false);
		}
		else if (args.length == 1)
		{
			new Coordinator().run(args[0], null, null, false);
		}
		else if (args.length == 2)
		{
			new Coordinator().run(args[0], args[1], null, false);
		}
		else if (args.length == 3)
		{
			new Coordinator().run(args[0], args[1], args[2], false);
		}
		else
		{
			System.out.println("wrong number of parameters: FinalYearProject.exe trainingfolder testfolder outputfolder");
		} 
	}
}