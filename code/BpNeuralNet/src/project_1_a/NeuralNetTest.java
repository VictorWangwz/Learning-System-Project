package project_1_a;

import java.io.File;
import java.io.IOException;

public class NeuralNetTest {

 public static void main(String[] args) throws IOException{
	 int[] layernum=new int[3];
	 layernum[0]=2;
	 layernum[1]=4;
	 layernum[2]=1;
	 double mobp=0.9;
	 double rate=0.2;
	 
	 double mobp1=0.0;
	 double rate1=0.2;
	 
	 final int epoch=10000;
	 double[][] in1={{0,0},{0,1},{1,0},{1,1}};
	 double[] out1={0,1,1,0};
	 double[] X1=new double[2];
	 
	 double[][] in2={{-1,-1},{-1,1},{1,-1},{1,1}};
	 double[] out2={-1,1,1,-1};
	 double[] X2=new double[2];
	 int countNum=10;
	 int sumBinary=0;
	 int aveBinary=0;
	 int maxBinary=0;
	 int minBinary=epoch;
	  for(int c=0;c<countNum;c++){
	 int i=0;
	double[] error1=new double[epoch];
		 BpNetWork myNet=new BpNetWork(layernum,rate,mobp,0,1);
		 myNet.initializeWeights();
	 for(;i<epoch;i++){
		for(int j=0;j<4;j++){
	    X1[0]=in1[j][0];
		X1[1]=in1[j][1];
		double argValue=out1[j];
		double trainOut=myNet.train(X1,argValue);
		error1[i]+=Math.pow(trainOut, 2)/2; 
		}
		//System.out.print(error1[i]);
		//System.out.print("\t");
		//System.out.print(error1[i]);
		if(error1[i]<0.05)
			break; 
		
		
	 }
	 if(i>maxBinary) maxBinary=i;
		if(i<minBinary) minBinary=i;
	 sumBinary+=i;
	 //System.out.print("\n");
	 //System.out.print(i);
	 }
	  aveBinary=sumBinary/countNum;
	  System.out.print("\n");
	 System.out.println("average number:"+aveBinary+"\tmax number:"+maxBinary+"\tmin number:"+minBinary);
	
	//System.out.println("\n");
	
	int countConv=0;
	 int maxBipolar=0;
	 int minBipolar=epoch;
	 int sumBipolar=0;
	 int aveBipolar=0;
	 for(int c=0;c<countNum;c++){
		 
		 BpNetWork myNet1=new BpNetWork(layernum,rate1,mobp1,-1,1);
		 myNet1.initializeWeights();
		 myNet1.argA=-1;
		 int m=0;
		 double[] error2=new double[epoch];
	 for(;m<epoch;m++){
		for(int n=0;n<4;n++){
	    X2[0]=in2[n][0];
		X2[1]=in2[n][1];
		double argValue2=out2[n];
		double trainOut2=myNet1.train(X2,argValue2);
		error2[m]+=Math.pow(trainOut2, 2)/2; 
		}
		//System.out.print(error2[m]);
		//System.out.print(error2[m]);
		//System.out.print("\t");
		
		if(error2[m]<0.05)
			{
			countConv++;
			break;
			
			}
		
	 }
	// if(m<epoch)
		// countConv++;
	 //System.out.print("\n");
	 if(m!=10000)
	 sumBipolar+=m;
	 if(m!=10000&&m>maxBipolar) maxBipolar=m;
		if(m<minBipolar) minBipolar=m;
	 //System.out.print(m);
	 //System.out.print("\n");
	 }
	 aveBipolar=sumBipolar/countNum;
     System.out.print("average number:"+aveBipolar+"\tmax number:"+maxBipolar+"\tmin number:"+minBipolar+"\tconverge time:"+countConv);
	
	 //myNet.save(new File("C:/Users/wangzhen/Desktop/EECE592/project_1_a/bin/weight.txt"));
	 
}
}
