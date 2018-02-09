package rlRobotNew;


import java.awt.*;   
import java.awt.geom.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

import NNRobot.Action;
import robocode.*;   
       
import robocode.AdvancedRobot;   
       
    public class newRobot1 extends AdvancedRobot   
    {   
      public static final double PI = Math.PI;   
      private Target target=new Target();   
      private QTable table;   
      private Learner learner;   
      private double reward = 0.0;   
      private double firePower;   
      private int direction = 1;   
      private int isHitWall = 0;   
      private int isHitByBullet = 0;   
      private boolean NNFlag=true;
      
     
        private static final double NN_alpha = 0.1; //learning rate for NN Q learning
    	private static final double NN_lambda = 0.9; //Discount rate for NN Q learning
    	private static double NN_epsilon = 0.0;
    	private int NN_last_action = 0;
    	private double[] NN_last_states = new double[5]; 
    	//e(s) statistics
      	private static double total_error_in_one_round = 0.0;
      	private static Vector<Double> error_statistics = new Vector<Double>(0);
      	
      	//Q-value learning history
      	private static Vector<Double> Q_value_history = new Vector<Double>(0);
      	private static final double test_heading = 100;
    	private static final double test_target_distance = 100;
    	private static final double test_target_bearing = 70;
    	private static final int test_action = 0;
    	private static final int test_hitWall = 0;
    	private static final int test_hitByBullet = 0;
    	private static double NN_test_heading = test_heading/180 -1;
    	private static double NN_test_target_distance = test_target_distance/500 -1;
    	private static double NN_test_target_bearing = test_target_bearing/180-1;
    	private static final double NN_test_hitWall = 0;
    	private static final double NN_test_hitByBullet = 0;
    	private static int NN_test_action = test_action;
      
      double rewardForWin=100;
      double rewardForDeath=-10;
      double accumuReward=0.0;
      private static int count=0;
      private static int countForWin=0;
      
      BpNetWork[] myNet=new BpNetWork[Action.NumRobotActions];

      public void run()   
      {   
        if(NNFlag==false){
        table = new QTable();   
        loadData();   
        learner = new Learner(table);   
        target = new Target();   
        target.distance = 1000;   
        }
        else
        {
        	
        	int[] layernum=new int[3];
   		    layernum[0]=5;
   		    layernum[1]=15;
   		    layernum[2]=1;
   		    double mobp=0.9;
   		    double rate=0.1;
   		    double[] maxminQ=new double[2];
   		    maxminQ[0]=1.0;
   		    maxminQ[1]=0.0;
        	//System.out.println(Action.NumRobotActions);
        	for(int i=0;i<Action.NumRobotActions;i++){
        		myNet[i]=new BpNetWork(layernum,rate,mobp,maxminQ[1],maxminQ[0]);
        		myNet[i].initializeWeights();
        		System.out.println("robotlayerweightis "+myNet[i].layerWeight.length+"\t"+myNet[i].layerWeight[0].length+"\t"+myNet[i].layerWeight[0][0].length+"\t");
        		try {
					myNet[i].loadWeight("C:/robocode/robots/rlRobotNew/newRobot1.data/NN_weights_from_LUT"+i+".txt");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
        	}
        	System.out.println(myNet[0].layerWeight[1][15][0]);
        }
    	
       
        setColors(Color.green, Color.white, Color.green);   
        setAdjustGunForRobotTurn(true);   
        setAdjustRadarForGunTurn(true);   
        turnRadarRightRadians(2 * PI);   
       /* if(getRoundNum()>500)
	    {
	    	  learner.explorationRate=0.3;
	     }
	    */
        while (true)   
        {   
          robotMovement();   
          firePower = 3000/target.distance;   
          if (firePower > 3)   
            firePower = 3;   
          radarMovement();   
          gunMovement();   
          if (getGunHeat() == 0) {   
            setFire(firePower);   
          }   
          execute();   

        }   
      }   
 
     
      private void robotMovement()   
      {  
    	int action;  
    	if(NNFlag==false){
        int state = getState();  
        action = learner.selectAction(state);   
     
        //learner.learn(state, action, reward);  
        learner.learnSARSA(state, action, reward);
        accumuReward+=reward;
    	}
    	else{
    		System.out.println(target.distance+"\t");
    		//System.out.println(target.bearing+"\t");
    	    //System.out.println(isHitWall+"\t");
    		//System.out.println(reward+"\t");				
    		//System.out.println(isHitByBullet+"\t");
    		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
    	}
       
        reward = 0.0;   
        isHitWall = 0;   
        isHitByBullet = 0;   
       
        switch (action)   
        {   
          case Action.RobotAhead:   
            setAhead(Action.RobotMoveDistance);   
            break;   
          case Action.RobotBack:   
            setBack(Action.RobotMoveDistance);   
            break;   
          case Action.RobotAheadTurnLeft:   
            setAhead(Action.RobotMoveDistance);   
            setTurnLeft(Action.RobotTurnDegree);      
            break;   
          case Action.RobotAheadTurnRight:   
            setAhead(Action.RobotMoveDistance);   
            setTurnRight(Action.RobotTurnDegree);    
            break;   
          case Action.RobotBackTurnLeft:   
            setAhead(Action.RobotMoveDistance);   
            setTurnRight(Action.RobotTurnDegree);    
            break;   
          case Action.RobotBackTurnRight:   
            setAhead(target.bearing);   
            setTurnLeft(Action.RobotTurnDegree);      
            break;   
        }   
      }    
       
      public int NeuralNetforAction(double heading, double targetDistance, double targetBearing, double isHitWall, double isHitByBullet, double reward)
      {
    	  int action = 0;
    	  double[]  NN_current_states= new double[5];
    	  NN_current_states[4]=heading;
    	  NN_current_states[3]=targetDistance;
    	  NN_current_states[2]=targetBearing;
    	  NN_current_states[1]=isHitWall;
    	  NN_current_states[0]=isHitByBullet;
    	  System.out.println(NN_current_states[0]+"\t"+NN_current_states[1]+"\t"+NN_current_states[2]+"\t"+NN_current_states[3]+"\t"+NN_current_states[4]+"\t");
    	  //get the best action
    	  for(int i=0; i<Action.NumRobotActions; i++)
    	  {
    		  if(myNet[i].outputFor(NN_current_states)>myNet[action].outputFor(NN_current_states))
    		  {
    			  action = i;
    		  }
    	  }
    	  
    	  //update weights
    	  double NN_Q_new=myNet[action].outputFor(NN_current_states);
    	  double error_signal = 0;
    	  //if(NN_last_states[0]==0.0||NN_last_states[1]==0.0);
    	 // else
    	  //{
    		  error_signal = NN_alpha*(reward + NN_lambda * NN_Q_new - myNet[NN_last_action].outputFor(NN_last_states));
    	  //}
    	  
    	  newRobot1.total_error_in_one_round += error_signal*error_signal/2;
    	  double correct_old_Q = myNet[NN_last_action].outputFor(NN_last_states) + error_signal;
    	  myNet[NN_last_action].train(NN_last_states, correct_old_Q); 
    	  if(Math.random() < NN_epsilon)
    	  {
    		  action = new Random().nextInt(Action.NumRobotActions);
    	  }
    	  
    	  for(int i=0; i<5; i++)
    	  {
    		  NN_last_states[i] = NN_current_states[i];
    	  }
    	  
    	  NN_last_action=action;
    	  System.out.println(target.distance+"\t");
    	  return action;
    	 
      }

	private int getState()   
      {   
        int heading = State.getHeading(getHeading());   
        int targetDistance = State.getTargetDistance(target.distance);   
        int targetBearing = State.getTargetBearing(target.bearing);   
        out.println("Stste(" + heading + ", " + targetDistance + ", " + targetBearing + ", " + isHitWall + ", " + isHitByBullet + ")");   
        int state = State.Mapping[heading][targetDistance][targetBearing][isHitWall][isHitByBullet];   
        return state;   
      }   
       
      private void radarMovement()   
      {   
        double radarOffset;   
        if (getTime() - target.ctime > 4) { //if we haven't seen anybody for a bit....   
          radarOffset = 4*PI;               //rotate the radar to find a target   
        } else {   
       
          //next is the amount we need to rotate the radar by to scan where the target is now   
          radarOffset = getRadarHeadingRadians() - (Math.PI/2 - Math.atan2(target.y - getY(),target.x - getX()));   
          //this adds or subtracts small amounts from the bearing for the radar to produce the wobbling   
          //and make sure we don't lose the target   
          radarOffset = NormaliseBearing(radarOffset);   
          if (radarOffset < 0)   
            radarOffset -= PI/10;   
          else   
            radarOffset += PI/10; 

        }   
        //turn the radar   
        setTurnRadarLeftRadians(radarOffset);   
      }   
       
      private void gunMovement()   
      {   
        long time;   
        long nextTime;   
        Point2D.Double p;   
        p = new Point2D.Double(target.x, target.y);   
        for (int i = 0; i < 20; i++)   
        {   
          nextTime = (int)Math.round((getrange(getX(),getY(),p.x,p.y)/(20-(3*firePower))));   
          time = getTime() + nextTime - 10;   
          p = target.guessPosition(time);   
        }   
        //offsets the gun by the angle to the next shot based on linear targeting provided by the enemy class   
        double gunOffset = getGunHeadingRadians() - (Math.PI/2 - Math.atan2(p.y - getY(),p.x -  getX()));   
        setTurnGunLeftRadians(NormaliseBearing(gunOffset));   
      }   

      //bearing is within the -pi to pi range   
      double NormaliseBearing(double ang){

        if (ang > PI)   
          ang -= 2*PI;   
        if (ang < -PI)    
          ang += 2*PI;   
        return ang;   
      }   
       
      //heading within the 0 to 2pi range   
      double NormaliseHeading(double ang) {   
        if (ang > 2*PI)   
          ang -= 2*PI;   
        if (ang < 0)   
          ang += 2*PI;   
        return ang;   
      }   
       
      //returns the distance between two x,y coordinates   
      public double getrange( double x1,double y1, double x2,double y2 )   
      {   
        double xo = x2-x1;   
        double yo = y2-y1;   
        double h = Math.sqrt( xo*xo + yo*yo );   
        return h;   
      }   
       
      //gets the absolute bearing between to x,y coordinates   
      public double absbearing( double x1,double y1, double x2,double y2 )   
      {  
        double xo = x2-x1;   
        double yo = y2-y1;   
        double h = getrange( x1,y1, x2,y2 );   
        if( xo > 0 && yo > 0 )   
        {   
          return Math.asin( xo / h );   
        }   
        if( xo > 0 && yo < 0 )   
        {   
          return Math.PI - Math.asin( xo / h );   
        }   
        if( xo < 0 && yo < 0 )   
        {   
          return Math.PI + Math.asin( -xo / h );   
        }   
        if( xo < 0 && yo > 0 )   
        {   
          return 2.0*Math.PI - Math.asin( -xo / h );   
        }   
        return 0;   
      }   
       
   
      
      public void onBulletHit(BulletHitEvent e)   
      {  

        if (target.name == e.getName())   
        {     
          double change = e.getBullet().getPower() * 9;   
          out.println("Bullet Hit: " + change);   
          accumuReward += change;  
          //int state = getState();   
          //int action = learner.selectAction(state);  
         // learner.learn(state, action, change);    
          //learner.learnSARSA(state, action, change);
          int action;
          if(NNFlag==false){
              int state = getState();   
              action = learner.selectAction(state);   
           
              //learner.learn(state, action, reward);  
              learner.learnSARSA(state, action, reward);
              accumuReward+=reward;
          	}
          	else{
          		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
          	}
        }   
      }   
       

      public void onBulletMissed(BulletMissedEvent e)   
      {   
        double change = -e.getBullet().getPower();   
        out.println("Bullet Missed: " + change);   
       
        accumuReward += change;  
        //int state = getState();   
       // int action = learner.selectAction(state);  
        //learner.learn(state, action, change);    
        //learner.learnSARSA(state, action, change);
        int action;
        if(NNFlag==false){
            int state = getState();   
            action = learner.selectAction(state);   
         
            //learner.learn(state, action, reward);  
            learner.learnSARSA(state, action, reward);
            accumuReward+=reward;
        	}
        	else{
        		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
        	}
      }   
       
      public void onHitByBullet(HitByBulletEvent e)   
      {   
        if (target.name == e.getName())   
        {   
          double power = e.getBullet().getPower();   
          double change = -(4 * power + 2 * (power - 1));   
          out.println("Hit By Bullet: " + change);   
          accumuReward += change;  
          //int state = getState();   
          int action;
          if(NNFlag==false){
              int state = getState();   
              action = learner.selectAction(state);   
           
              //learner.learn(state, action, reward);  
              learner.learnSARSA(state, action, reward);
              accumuReward+=reward;
          	}
          	else{
          		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
          	}
          //int action = learner.selectAction(state);  
          //learner.learn(state, action, change); 
          //learner.learnSARSA(state, action, change);
        }   
        isHitByBullet = 1;   
      }   
       
      public void onHitRobot(HitRobotEvent e)   
      {   
        if (target.name == e.getName())   
        {   
          double change = -6.0;   
          out.println("Hit Robot: " + change);   
          accumuReward += change;  
          //int state = getState();   
          //int action = learner.selectAction(state);  
          //learner.learn(state, action, change);   
          //learner.learnSARSA(state, action, change);
          int action;
          if(NNFlag==false){
              int state = getState();   
              action = learner.selectAction(state);   
           
              //learner.learn(state, action, reward);  
              learner.learnSARSA(state, action, reward);
              accumuReward+=reward;
          	}
          	else{
          		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
          	}
        }   
      }   
       
      public void onHitWall(HitWallEvent e)   
      {   
           
        double change = -(Math.abs(getVelocity()) * 0.5 );   
        out.println("Hit Wall: " + change);   
        accumuReward += change;  
        //int state = getState();   
        //int action = learner.selectAction(state);  
        //learner.learn(state, action, change);   
        //learner.learnSARSA(state, action, change);
        int action;
        if(NNFlag==false){
            int state = getState();   
            action = learner.selectAction(state);   
         
            //learner.learn(state, action, reward);  
            learner.learnSARSA(state, action, reward);
            accumuReward+=reward;
        	}
        	else{
        		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
        	}
        isHitWall = 1;   
      }   
      

      public void onScannedRobot(ScannedRobotEvent e)   
      {   
        if ((e.getDistance() < target.distance)||(target.name == e.getName()))   
        {   
          //the next line gets the absolute bearing to the point where the bot is   
          double absbearing_rad = (getHeadingRadians()+e.getBearingRadians())%(2*PI);   
          //this section sets all the information about our target   
          target.name = e.getName();   
          double h = NormaliseBearing(e.getHeadingRadians() - target.head);   
          h = h/(getTime() - target.ctime);   
          target.changehead = h;   
          target.x = getX()+Math.sin(absbearing_rad)*e.getDistance(); //works out the x coordinate of where the target is   
          target.y = getY()+Math.cos(absbearing_rad)*e.getDistance(); //works out the y coordinate of where the target is   
          target.bearing = e.getBearingRadians();   
          target.head = e.getHeadingRadians();   
          target.ctime = getTime();             //game time at which this scan was produced   
          target.speed = e.getVelocity();   
          target.distance = e.getDistance();   
          target.energy = e.getEnergy();   
        }   
      }   
       
      public void onRobotDeath(RobotDeathEvent e)   
      {   
       
        if (e.getName() == target.name)   
        {
        	target.distance = 1000; 
        }
         
      }   
       
      public void onWin(WinEvent event)   
      {   
    	  File file0 = getDataFile("NN_weights_from_LUT0.dat"); 
    	  File file1 = getDataFile("NN_weights_from_LUT0.dat"); 
    	  File file2 = getDataFile("NN_weights_from_LUT0.dat"); 
    	  File file3 = getDataFile("NN_weights_from_LUT0.dat"); 
    	  File file4 = getDataFile("NN_weights_from_LUT0.dat"); 
    	  try {
			save(file0,myNet[0]);
			save(file1,myNet[1]);
			save(file2,myNet[2]);
			save(file3,myNet[3]);
			save(file4,myNet[4]);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    	  /*for(int j=0;j<Action.NumRobotActions;j++){
    			 try {
					myNet[j].save(new File("C:/robocode/robots/rlRobotNew/newRobot1.data/NN_weights_from_LUT"+j+".txt"));
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
    			 }*/
    File file = getDataFile("accumReward.dat");
    accumuReward+=rewardForWin;
    int action;
    if(NNFlag==false){
        int state = getState();   
        action = learner.selectAction(state);   
     
        //learner.learn(state, action, reward);  
        learner.learnSARSA(state, action, reward);
        accumuReward+=reward;
    	}
    	else{
    		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
    	}
    //int state = getState();   
    //int action = learner.selectAction(state);  
    //learner.learn(state, action, rewardForWin);  
    //learner.learnSARSA(state, action, rewardForWin);
        robotMovement();
        saveData(); 
		 int winningFlag=7;
		 countForWin++;
		 count++;
		 PrintStream w = null; 
		    try 
		    { 
		      w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
		      if(count==50){
					 count=0;
		      w.println(accumuReward+" "+countForWin*2+"\t"+winningFlag+" "+" "+learner.explorationRate); 
		      accumuReward=0; 
			     countForWin=0;
		      if (w.checkError()) 
		        System.out.println("Could not save the data!");  //setTurnLeft(180 - (target.bearing + 90 - 30));
		      w.close(); 
		      }
		    } 
		    catch (IOException e) 
		    { 
		      System.out.println("IOException trying to write: " + e); 
		    } 
		    finally 
		    { 
		      try 
		      { 
		        if (w != null) 
		          w.close(); 
		      } 
		      catch (Exception e) 
		      { 
		        System.out.println("Exception trying to close witer: " + e); 
		      } 

		    } 
      }   
      public void save(File file,BpNetWork myNet) throws IOException {

  		 PrintStream w = null; 
  		    try 
  		    { 
  		      w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
  		    
			//for(int i = 0; i<this.layerWeight.length; i++)
			//{
			    int i=0;
				
				//System.out.println(this.layerWeight[i].length);
				
				for(int j=0;j<myNet.layerWeight[i].length;j++){
					//System.out.println(this.layerWeight[i][j].length);
					//System.out.println(this.layerWeight[i][j][0]);
					for(int k=0;k<myNet.layerWeight[i][j].length;k++)
						{
						w.println(myNet.layerWeight[i][j][k]);
						
						}
					
					
				}
				//if(this.writeWeights(weight_vector,argFile) == 0)
					//System.out.println("Writing hidden nodes fails");
				
				i++;
				for(int j=0;j<myNet.layerWeight[i].length;j++){
					//System.out.println(this.layerWeight[i][j].length);
					//System.out.println(this.layerWeight[i][j][0]);
					w.println(myNet.layerWeight[i][j][0]);
					
					
				}
				 if (w.checkError()) 
				        System.out.println("Could not save the data!"); 
				      w.close(); 
				      
			//}
  		     
  		    } 
  		    catch (IOException e) 
  		    { 
  		      System.out.println("IOException trying to write: " + e); 
  		    } 
  		    finally 
  		    { 
  		      try 
  		      { 
  		        if (w != null) 
  		          w.close(); 
  		      } 
  		      catch (Exception e) 
  		      { 
  		        System.out.println("Exception trying to close witer: " + e); 
  		      } 

  		    }
			
			
			
		}
      public void onDeath(DeathEvent event)   
      {   
    	  File file0 = getDataFile("NN_weights_from_LUT0.dat"); 
    	  File file1 = getDataFile("NN_weights_from_LUT1.dat"); 
    	  File file2 = getDataFile("NN_weights_from_LUT2.dat"); 
    	  File file3 = getDataFile("NN_weights_from_LUT3.dat"); 
    	  File file4 = getDataFile("NN_weights_from_LUT4.dat"); 
    	  try {
			save(file0,myNet[0]);
			save(file1,myNet[1]);
			save(file2,myNet[2]);
			save(file3,myNet[3]);
			save(file4,myNet[4]);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    	  accumuReward+=rewardForDeath;
    	  int action;
    	    if(NNFlag==false){
    	        int state = getState();   
    	        action = learner.selectAction(state);   
    	     
    	        //learner.learn(state, action, reward);  
    	        learner.learnSARSA(state, action, reward);
    	        accumuReward+=reward;
    	    	}
    	    	else{
    	    		action=this.NeuralNetforAction(getHeading()/180-1,target.distance/500-1,target.bearing/180-1,isHitWall*2-1,isHitByBullet*2-1, reward);
    	    	}
    	  //int state = getState();   
          //int action = learner.selectAction(state);  
         // learner.learn(state, action, rewardForDeath); 
          //learner.learnSARSA(state, action, rewardForDeath);
          count++;
      	
        saveData();   
        File file = getDataFile("accumReward.dat"); 
        int losingFlag=5;
		 PrintStream w = null; 
		    try 
		    { 
		      w = new PrintStream(new RobocodeFileOutputStream(file.getAbsolutePath(), true)); 
		      if(count==50){
			    	count=0;
		      w.println(accumuReward+" "+countForWin*2+"\t"+losingFlag+" "+" "+learner.explorationRate); 
		      accumuReward=0;
		      countForWin=0;
		      if (w.checkError()) 
		        System.out.println("Could not save the data!"); 
		      w.close(); 
		      }
		    } 
		    catch (IOException e) 
		    { 
		      System.out.println("IOException trying to write: " + e); 
		    } 
		    finally 
		    { 
		      try 
		      { 
		        if (w != null) 
		          w.close(); 
		      } 
		      catch (Exception e) 
		      { 
		        System.out.println("Exception trying to close witer: " + e); 
		      } 

		    } 
      }   
       
      public void loadData()   
      {   
        try   
        {   
          table.loadData(getDataFile("movement.dat"));   
        }   
        catch (Exception e)   
        {   
        }   
      }   
       
      public void saveData()   
      {   
        try   
        {   
          table.saveData(getDataFile("movement.dat"));   
        }   
        catch (Exception e)   
        {   
          out.println("Exception trying to write: " + e);   
        }   
      }   
    }  
