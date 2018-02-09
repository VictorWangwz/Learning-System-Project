package rlRobotNew;




import java.io.*;
import java.util.ArrayList;

import robocode.DeathEvent;
import robocode.RobocodeFileOutputStream;
import robocode.*;   

import robocode.AdvancedRobot;   

public class BpNetWork implements NeuralNetInterface {
	
	int argNumInputs;
	int argNumHidden;
	double argLearningRate;
	double argMomentumTerm;
	double argA;
	double argB ;
	public double[][] layer;//神经网络各层节点
    public double[][] layError;//神经网络各节点误差
    public double[][][] layerWeight;//各层节点权重
    public double[][][] layerWDelta;//各层节点权重动量
    
	//double inputValue[]=new double[argNumInputs+1];
	BpNetWork(){
		argNumInputs=5;
		argNumHidden=15;
		argLearningRate=0.1;
		argMomentumTerm=0.9;
		argA=0;
		argB=1;
		layer = new double[3][];
        layError = new double[3][];
        layerWeight = new double[2][][];
        layerWDelta = new double[2][][];
		/*for(int i=0;i<inputValue.length-1;i++){
			inputValue[i]=0;
		}
		inputValue[inputValue.length-1]=1;*/
	}
	BpNetWork(int[] layernum, double rate, double mobp,double A,double B)
	{
		argNumInputs=layernum[0];
		argNumHidden=layernum[1];
		argA=A;
		argB=B;
		    this.argMomentumTerm = mobp;
	        this.argLearningRate = rate;
	        layer = new double[layernum.length][];
	        layError = new double[layernum.length][];
	        layerWeight = new double[layernum.length-1][][];
	        layerWDelta = new double[layernum.length-1][][];
	}
	
	 int NumInputWBias=argNumInputs+1; 
	 int NumHidWBias=argNumHidden+1;
	 
	
	 /**
		 * Return a bipolar sigmoid of the input X
		 * @param x The input
		 * @return f(x) = 2 / (1+e(-x)) - 1
		 */
	 public double sigmoid(double x){
		 return 2/(1+Math.pow(Math.E, -x))-1;
	 };
	

	 /**
	 * This method implements a general sigmoid with asymptotes bounded by (a,b)
	 * @param x The input
	 * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
	 */
	 public double customSigmoid(double x){
		 return (argB-argA)/(1+Math.pow(Math.E, -x))+argA;
	 };

	 /**
	 * Initialize the weights to random values.
	 * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
	 * Like wise for hidden units. For say 2 hidden units which are stored in an array.
	 * [0] & [1] are the hidden & [2] the bias.
	 * We also initialise the last weight change arrays. This is to implement the alpha term.
	 */
	 public void initializeWeights(){
		 int[] layerNum=new int[3];
		 layerNum[0]=this.argNumInputs;
		 layerNum[1]=this.argNumHidden;
		 layerNum[2]=1;
		 for(int l=0;l<layerNum.length;l++){
	            layer[l]=new double[layerNum[l]];
	            layError[l]=new double[layerNum[l]];
	            if(l+1<layerNum.length){
	                layerWeight[l]=new double[layerNum[l]+1][layerNum[l+1]];
	                layerWDelta[l]=new double[layerNum[l]+1][layerNum[l+1]];
	                for(int j=0;j<layerNum[l]+1;j++)
	                    for(int i=0;i<layerNum[l+1];i++)
	                        layerWeight[l][j][i]=Math.random()-0.5;
	            }   
	        }
	 }


	 /**
	 * Initialize the weights to 0.
	 */
	 
	 public void zeroWeights(){ 
	int[] layerNum=new int[3];
	 layerNum[0]=this.argNumInputs;
	 layerNum[1]=this.argNumHidden;
	 layerNum[2]=1;
	
	 for(int l=0;l<layerNum.length;l++){
            layer[l]=new double[layerNum[l]];
            layError[l]=new double[layerNum[l]];
            if(l+1<layerNum.length){
                layerWeight[l]=new double[layerNum[l]+1][layerNum[l+1]];
                layerWDelta[l]=new double[layerNum[l]+1][layerNum[l+1]];
                for(int j=0;j<layerNum[l]+1;j++)
                    for(int i=0;i<layerNum[l+1];i++)
                        layerWeight[l][j][i]=0;
            }   
        }
 }

	 
	 
	 
	 ////commone interface part
	 /**
	  * @param X The input vector. An array of doubles.
	  * @return The value returned by th LUT or NN for this input vector
	  */
	  public double outputFor(double [] X){
		   for(int l=1;l<layer.length;l++){
	            for(int j=0;j<layer[l].length;j++){
	                double z=layerWeight[l-1][layer[l-1].length][j];
	                for(int i=0;i<layer[l-1].length;i++){
	                    layer[l-1][i]=l==1?X[i]:layer[l-1][i];
	                    z+=layerWeight[l-1][i][j]*layer[l-1][i];
	                }
	                layer[l][j]=this.customSigmoid(z);
	            }
	        }
		  
		  return layer[layer.length-1][0];
	  }



	  /**
	  * This method will tell the NN or the LUT the output
	  * value that should be mapped to the given input vector. I.e.
	  * the desired correct output value for an input.
	  * @param X The input vector
	 * @param argValue The new value to learn
	 * @return The error in the output for that input vector
	 */
	 public double train(double [] X, double argValue){
	 double out=outputFor(X);
	 double[] val=new double[1];
	 val[0]=argValue;
	
	  int l=layer.length-1;
	 // System.out.print(l);
      for(int j=0;j<layError[2].length;j++)
          layError[2][j]=(val[j]-out)*1/(this.argB-this.argA)*(out-argA)*(argB-out);
      
     /* l--;

      for(;l>=0;l--){
          for(int j=0;j<layError[l].length;j++){
              double z = 0.0;
              for(int i=0;i<layError[l+1].length;i++){
                  z=z+l>0?layError[l+1][i]*layerWeight[l][j][i]:0;
                  layerWDelta[l][j][i]= this.argMomentumTerm*layerWDelta[l][j][i]+this.argLearningRate*layError[l+1][i]*layer[l][j];//隐含层动量调整
                  layerWeight[l][j][i]+=layerWDelta[l][j][i];
                  if(j==layError[l].length-1){
                      layerWDelta[l][j+1][i]= this.argMomentumTerm*layerWDelta[l][j+1][i]+this.argLearningRate*layError[l+1][i];//截距动量调整
                      layerWeight[l][j+1][i]+=layerWDelta[l][j+1][i];
                  }
              }
              layError[l][j]=z*1/(this.argB-this.argA)*(layer[l][j]-argA)*(argB-layer[l][j]);//记录误差
          }
      
      }*/
     
      //hid-output 
      for(int j=0;j<this.argNumHidden;j++){
    	  double preW=layerWeight[1][j][0];
    	  layerWeight[1][j][0]+=this.argMomentumTerm*layerWDelta[1][j][0]+this.argLearningRate*layError[2][0]*layer[1][j];
    	  layerWDelta[1][j][0]=layerWeight[1][j][0]-preW;
      }
      layerWDelta[1][this.argNumHidden][0]=this.argMomentumTerm*layerWDelta[1][this.argNumHidden][0]+this.argLearningRate*layError[2][0];
	  layerWeight[1][this.argNumHidden][0]+=layerWDelta[1][this.argNumHidden][0];
	  
      
      //input-hidden
      for(int j=0;j<this.argNumHidden;j++){
    	  layError[1][j]=layError[2][0]*1/(this.argB-this.argA)*(layer[1][j]-argA)*(argB-layer[1][j])*layerWeight[1][j][0];
    	  
    	  for(int i=0;i<this.argNumInputs;i++){
    		  layerWDelta[0][i][j]=this.argMomentumTerm*layerWDelta[0][i][j]+this.argLearningRate*layError[1][j]*layer[0][i];
    		  layerWeight[0][i][j]+=layerWDelta[0][i][j];
    	  }
    	  layerWDelta[0][this.argNumInputs][j]=this.argMomentumTerm*layerWDelta[0][this.argNumInputs][j]+this.argLearningRate*layError[1][j];
		  layerWeight[0][this.argNumInputs][j]+=layerWDelta[0][this.argNumInputs][j];
    	  
      }
      
      //System.out.print(layer_weight);
	 return val[0]-out;
}


	  /**
	  * A method to write either a LUT or weights of an neural net to a file.
	  * @param argFile of type File.
	 * @throws FileNotFoundException 
	 * @throws IOException 
	  */
	 public void save(File argFile) throws IOException {
			//File weightsfile = new File("weightsfile.lqn.txt");
			if(argFile.exists())
			{
				FileWriter erasor = new FileWriter(argFile);
				erasor.write(new String());
				erasor.close();
			}
			else
			{
				argFile.createNewFile();
			}
			
			int count=0;
			//for(int i = 0; i<this.layerWeight.length; i++)
			//{
			    int i=0;
				ArrayList<double[]> weight_vector = new ArrayList<double[]>(0);
				//System.out.println(this.layerWeight[i].length);
				
				for(int j=0;j<this.layerWeight[i].length;j++){
					double[] buffer=new double[this.layerWeight[i][j].length];
					//System.out.println(this.layerWeight[i][j].length);
					//System.out.println(this.layerWeight[i][j][0]);
					for(int k=0;k<this.layerWeight[i][j].length;k++)
						{
						buffer[k]=layerWeight[i][j][k];
						double[] bufferN=new double[1];
						bufferN[0]=buffer[k];
						weight_vector.add(count++, bufferN);
						}
					
					
				}
				//if(this.writeWeights(weight_vector,argFile) == 0)
					//System.out.println("Writing hidden nodes fails");
				
				i++;
				for(int j=0;j<this.layerWeight[i].length;j++){
					double[] buffer=new double[1];
					//System.out.println(this.layerWeight[i][j].length);
					//System.out.println(this.layerWeight[i][j][0]);
					buffer[0]=layerWeight[i][j][0];
					
					weight_vector.add(count++, buffer);
				}
				if(this.writeWeights(weight_vector,argFile) == 0)
					System.out.println("Writing output nodes fails");
			//}
			
		}

	 public int writeWeights(ArrayList<double[]> content, File file) throws IOException
		{
			BufferedWriter bw = new BufferedWriter (new FileWriter(file,true));
			for(int i=0; i<content.size(); i++)
			{
				for(int j = 0; j<content.get(i).length;j++)
				{
					bw.write(content.get(i)[j]+"\t" );
					bw.flush();
				}
				bw.newLine();
				bw.flush();
			}
			bw.write("============================================");
			bw.newLine();
			bw.flush();
			bw.close();
			content.clear();
			System.out.println("Writing successed");
			return 1;
		}	 
	 
	 /**
		 * Loads the weights from file. Format of the file is expected to follow
		 * that specified in the "save" method specified elsewhere in this class.
		 * @param argFileName the name of the file where the weights are to be found
		 */
		public void loadWeight ( String argFileName ) throws IOException {

			FileInputStream inputFile = new FileInputStream( argFileName );
			BufferedReader inputReader = new BufferedReader(new InputStreamReader( inputFile ));		
					
			
			// First load the weights from the input to hidden neurons (one line per weight)
			//System.out.println(this.layerWeight[0].length+"\n");
			for ( int i=0; i<this.layerWeight[0].length; i++) {
				for(int j=0;j<this.layerWeight[0][i].length;j++)
					this.layerWeight[0][i][j]=Double.valueOf( inputReader.readLine() );
			
			}
			for(int i=0;i<this.layerWeight[1].length;i++){
				this.layerWeight[1][i][0] = Double.valueOf( inputReader.readLine() );
			}
			inputReader.close();
			
		}	

	 

	  /**
	  * Loads the LUT or neural net weights from file. The load must of course
	  * have knowledge of how the data was written out by the save method.
	  * You should raise an error in the case that an attempt is being
	  * made to load data into an LUT or neural net whose structure does not match
	  * the data in the file. (e.g. wrong number of hidden neurons).
	  * @throws IOException
	  */
	 
	  public void load(String argFileName) throws IOException{
		 FileInputStream fin;
		 try{
			 fin = new FileInputStream(argFileName);
		 }catch(FileNotFoundException exc){
			 System.out.println("file not found");
			 return;
		 }

		 int[] layernum=new int[3];
		 layernum[0]=this.argNumInputs;
		 layernum[1]=this.argNumHidden;
		 layernum[2]=1;
		
		 for(int l=0;l<layernum.length;l++){
	            layer[l]=new double[layernum[l]];
	            layError[l]=new double[layernum[l]];
	            if(l+1<layernum.length){
	                layerWeight[l]=new double[layernum[l]+1][layernum[l+1]];
	                layerWDelta[l]=new double[layernum[l]+1][layernum[l+1]];
	                for(int j=0;j<layernum[l]+1;j++)
	                    for(int i=0;i<layernum[l+1];i++)
	                        layerWeight[l][j][i]=fin.read();
	            }   
	        }
		  fin.close();
	  }
	  
	  
}
