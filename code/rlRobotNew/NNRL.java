package rlRobotNew;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import NNRobot.Action;




public class NNRL {
	 public static final int NumHeading = 4;  
	 public static final int NumTargetDistance = 10;  
	 public static final int NumTargetBearing = 4;  
	 public static final int NumHitWall = 2;  
	 public static final int NumHitByBullet = 2;  
	 
	 public static final int NumActions = 6;  
	 
	 public static final int NumSpace=NumHeading*NumTargetDistance*NumTargetBearing*NumHitWall*NumHitByBullet*NumActions;
	 
	 //Load look-up table
	 private static double[] table_loaded = new double[NumSpace];
	 
	 //Process the look-up table
	 private static double[] table_processed = new double[NumSpace];
	 
	 //learning rate and momentum
	 public static final double learningRate = 0.1;   
	 
	 public static final double momentum = 0.9;
	 
	 //threshold
	 public static final double threshold = 0.01;
	 
	 //Initialization
	 private void initialize()
	 {
		 for(int i=0; i< NumSpace; i++)
		 {
			 table_loaded[i]=0;
			 table_processed[i]=0;
		 }
	 }
	 public static void main(String[] args) throws IOException{
		 int[] layernum=new int[3];
		 layernum[0]=5;
		 layernum[1]=15;
		 layernum[2]=1;
		 double mobp=0.9;
		 double rate=0.01;
		 //loadData("C:/robocode/robots/rlRobotNew/newRobot1.data/movement.dat");
		 load_table("C:/robocode/robots/rlRobot/newRobot1.data/movement.dat");
		 process_table();//set 1 for the action with highest Q value for each state
		 
		 double[][] stateInput = new double [NumSpace][5];
		 int StateNum=0;
		    for (int a = 0; a < NumHeading; a++)
		        for (int b = 0; b < NumTargetDistance; b++)
		          for (int c = 0; c < NumTargetBearing; c++)
		            for (int d = 0; d < NumHitWall; d++)
		              for (int e = 0; e < NumHitByBullet; e++) {
		            	  /*
		            		  stateInput[StateNum][0]=(e+1)/2;
		            		  stateInput[StateNum][1]=(d+1)/2;
		            		  stateInput[StateNum][2]=(c+1)/4;
		            		  stateInput[StateNum][3]=(b+1)/20;
		            		  stateInput[StateNum][4]=(a+1)/4;*/
		            	  stateInput[StateNum][0]=e;
		            	  stateInput[StateNum][0]/=2.0;
	            		  stateInput[StateNum][1]=d;
	            		  stateInput[StateNum][1]/=2.0;
	            		  stateInput[StateNum][2]=c;
	            		  stateInput[StateNum][2]/=4.0;
	            		  stateInput[StateNum][3]=b;
	            		  stateInput[StateNum][3]/=10.0;
	            		  stateInput[StateNum][4]=a;
	            		  stateInput[StateNum][4]/=4.0;
		            	      StateNum=StateNum+1;
		              }
		 //build network with multiple nets.
	
		 BpNetWork[] myNet=new BpNetWork[NumActions];
		 double[] maxminQ=new double[2];
		 maxminQ=findMaxMinQ(table_processed);
		 //maxminQ=findMaxMinQ(table_loaded);
		 //System.out.println("maxQ:"+maxminQ[0]+"minQ:"+maxminQ[1]+"\n");
		 for(int i=0;i<NumActions;i++){
			 myNet[i]=new BpNetWork(layernum,rate,mobp,maxminQ[1],maxminQ[0]);
			 myNet[i].initializeWeights();
		 }
		
		 final int epoch=10000;
		 
		//double[] inp=new double[]{1,2,3,4,5,6};
		//double out=1;
		//myNet.train(inp, out);
		//Training
		 double total_error[] = new double[]{0,0,0,0,0,0};
		 double error = 0;
		 double max_error = 0.0;
		 int iteration=0;
		 //System.out.println(NumSpace/NumActions);
		 
		 for(;iteration<epoch;){
			 max_error=0.0;
			 double totalErr=0;
			 for(int j=0;j<NumActions;j++){
				 total_error[j]=0;
			 for(int i=0; i<NumSpace/NumActions;i++)//NumSpace/NumActions; i++)
			 {
				
					 double[] inputArray;
					 inputArray=new double[5];
					 //for(int k=0;k<5;k++){
						 inputArray[0]=stateInput[i*NumActions][0];
						 inputArray[1]=stateInput[i*NumActions][1];
						 inputArray[2]=stateInput[i*NumActions][2];
						 inputArray[3]=stateInput[i*NumActions][3];
						 inputArray[4]=stateInput[i*NumActions][4];
					// System.out.println(inputArray[0]+"\t"+inputArray[1]+"\t"+inputArray[2]+"\t"+inputArray[3]+"\t"+inputArray[4]+"\t");
					 //inputArray=generateInputVector(generateInputandActionFromTable(i*NumActions));
					 double outputExp=generateCorrectOutput(i*NumActions,j);
					 //System.out.println(outputExp+"\n");
					 error = Math.pow(myNet[j].train(inputArray,outputExp), 2);
				     total_error[j] += error; 
				     if(max_error<Math.abs(error))
				    	 max_error=Math.abs(error);
				 }
			 //System.out.println("total_error["+j+"]:"+ total_error[j]+"\n");
			 }
			 
			 iteration++;
			 for(int m=0;m<6;m++){
				 total_error[m]=Math.sqrt(total_error[m]/NumSpace);
				 totalErr+=total_error[m];
				 
			 }
			 System.out.println(iteration+" total_error "+ totalErr+'\t'+total_error[0]+'\t'+total_error[1]+'\t'+total_error[2]+'\t'+total_error[3]+'\t'+total_error[4]+'\n'); 
			 
			 if(max_error < threshold) break;
		 }
	 
		 
		 for(int j=0;j<NumActions;j++){
		 myNet[j].save(new File("C:/robocode/robots/rlRobotNew/newRobot1.data/NN_weights_from_LUT"+j+".txt"));
		 }
		 /*for(int i=0;i<Action.NumRobotActions;i++){
     		//myNet[i]=new BpNetWork();
     		try {
					myNet[i].loadWeight("C:/robocode/robots/rlRobotNew/newRobot1.data/NN_weights_from_LUT"+i+".txt");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
     	}*/
		 	 
		 //System.out.println(myNet[0].layerWeight[0][0][0]+"\t"+myNet[0].layerWeight[1][0][0]+"\t"+myNet[0].layerWeight[1][15][0]);
		 
		 /*
		 //build one network
		 
		 BpNetWork myNet=new BpNetWork(layernum,rate,mobp,0,1);
		 double[] maxminQ=new double[2];
		 maxminQ=findMaxMinQ(table_loaded);
		 System.out.println("maxQ:"+maxminQ[0]+"minQ:"+maxminQ[1]+"\n");
		
		 final int epoch=10;
		//double[] inp=new double[]{1,2,3,4,5,6};
		//double out=1;
		//myNet.train(inp, out);
		//Training
		 double total_error = 0;
		 double error = 0;
		 double max_error = 0.0;
		 int iteration=0;
		 System.out.println(NumSpace/NumActions);
		 for(;iteration<epoch;){
			 max_error=0.0;
			 
			 for(int i=0; i<NumSpace; i++)
			 {
				
					 double[] inputArray=new double[5];
					 inputArray=generateInputandActionFromTable(i);
					 double outputExp=generateCorrectOutput(i,i%NumActions);
					 error = Math.pow(myNet.train(inputArray,outputExp), 2)/2;
				     total_error += error; 
				     if(max_error<Math.abs(error))
				    	 max_error=Math.abs(error);
				 }
			 //System.out.println("total_error["+j+"]:"+ total_error[j]+"\n");
			 
		     iteration++;
			 System.out.println(" max_error "+ max_error+" iteration "+iteration); 
			 if(max_error < threshold) 
				 break;
		 }
			 
			 
		 }
	 */
		 
}


		private static double[] findMaxMinQ(double[] table) {
		double[] maxminQ=new double[2];
		maxminQ[0]=Integer.MIN_VALUE;
		maxminQ[1]=Integer.MAX_VALUE;
			for(int i=0;i<table.length;i++){
			if(table[i]>maxminQ[0])
				maxminQ[0]=table[i];
			if(table[i]<maxminQ[1])
				maxminQ[1]=table[i];
			//if(table[i]!=0)
			//System.out.println(table[i]+"\n");
		}
			//System.out.println(table.length+"\n");
			return maxminQ;
	}
		//compute error
		 public static double computeError(double[] errorFromOutput)
		 {
			 double total_error = 0;
			 for(int i = 0; i < errorFromOutput.length; i++)
			 {
				 total_error += errorFromOutput[i];
			 }
			 
			 return total_error;
		 }
		 
		 //load table
		 public static void load_table(String filename) throws IOException
		 {
			 File readFile = new File(filename);
			 BufferedReader br = new BufferedReader(new FileReader(readFile));
			 String str;
			 int count = 0;
			 
			 while((str = br.readLine())!= null)
			 {
				 if(count < NumSpace)
				 {
					 table_loaded[count]=Double.parseDouble(str);
					 count++;
				 }
				 else
				 {
					 break;
				 }
			 }
			 
			 br.close();
		 }
		 
		 public static void process_table()
		 {
			 for(int i=0; i<NumSpace; i=i+NumActions)
			 {
				 int max = 0;
				 
				 for(int j=0; j<NumActions; j++)
				 {
					 if(table_loaded[i+j]<table_loaded[i+max])
					 {
						 table_processed[i+j] = 0.0;
					 }
					 else
					 {
						 table_processed[i+max] = 0.0;
						 table_processed[i+j] = 1.0;
						 max = j;
					 }
				 }
			 }
		 }
		 
		 //There is an action within return array
		 public static double[] generateInputandActionFromTable(int index)
		 {
			 int heading = index/(NumTargetDistance*NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
			 int left = index % (NumTargetDistance*NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
			 int targetDistances = left/(NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
			 left = left % (NumTargetBearing*NumHitWall*NumHitByBullet*NumActions);
			 int targetBearing = left/(NumHitWall*NumHitByBullet*NumActions);
			 left = left % (NumHitWall*NumHitByBullet*NumActions);
			 int hitWall = left/(NumHitByBullet*NumActions);
			 left = left % (NumHitByBullet*NumActions);
			 int hitByBullet = left % NumActions;
			 int action = left;  //This action might be wrong if index is not times of Action
					 
			 double[] return_array = new double[6];
			 
			 return_array[0]=heading;
			 return_array[1]=targetDistances;
			 return_array[2]=targetBearing;
			 return_array[3]=hitWall;
			 return_array[4]=hitByBullet;
			 return_array[5]=action;
			 
			 return return_array;
		 }
		 
		 public static double[] generateInputVector(double [] table_array)
		 {
			 if(table_array.length !=6)
			 {
				 System.out.println("Wrong Array Length");
			 }
			 
			 double[] return_array = new double[5];
			 return_array[0] = table_array[0];
			 return_array[1] = table_array[1];
			 return_array[2] = table_array[2];
			 return_array[3] = table_array[3];
			 return_array[4] = table_array[4];
			 
			 return return_array;
			 
		 }
		 
		 public static double generateCorrectOutput(int index,int j)
		 //for all combination of action and state as index (input of NN), get the 
		 //correctOutput as whether this state+action is chosen or not with the table _processed (==1 means the Q is the highest and it is the chosen one)
		 {
			 int state_index = index - index%NumActions;
			 double correctOutput;
			 //for(int i=0; i<NumActions; i++)
			 
			correctOutput=table_processed[state_index + j];
			//correctOutput=table_loaded[state_index + j];
			 
			 return correctOutput;
		 }
		 
	
	
	
	
	
}
