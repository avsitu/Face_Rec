import java.io.File;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.Scanner;
import java.lang.String;
import java.lang.Math;

public class face_rec
{
    public static int inputs = 120*128+1; //number of inputs
    public static int hiddenLayer = 15; //number of hidden layer nodes
    public static double learningRate = 0.2, targetM = 0.8, targetF = 0.2;
    public static double opOut, totalError, target, opBias;

    public static double[][] hdWeight = new double[hiddenLayer][inputs];   //weight table for inputs to 3 greyscale hiddenlayer
    public static double[] opWeight = new double[hiddenLayer];  //weight table for hiddenlayer to single output
    public static double[] pixels = new double[inputs];
    public static double[] hdOut = new double[hiddenLayer];
    public static double[] tempW = new double[hiddenLayer];
    public static String[][] fileBlock = new String[5][55];
    public static String[] testBlock = new String[40];
    
    public static void main(String[] args)
    {
        try
        {
            //partition and store file data into fileBlock as strings
            partition();

            if(args[0].equalsIgnoreCase("-train"))
            {
                int trial1[] = {0,1,2,3}, trial2[] = {1,2,3,4}, trial3[] = {2,3,4,0};
                int trial4[] = {3,4,0,1}, trial5[] = {4,0,1,2};
                double testAccu[] = new double[5];
                double trainAccu[] = new double[5];

                System.out.println("-----Cross Validation Training----- \n");
                //10 times of 5-fold cross validation
                for(int run = 1; run <= 10; run++) {
                    double testSum = 0, trainSum = 0;
                    initWeight();
                    System.out.println("Cross Test 1 of Run " + run);
                    for(int i : trial1)                             
                        train(fileBlock[i]);
                    totalError = 0.5*(target - opOut)*(target - opOut);
                    //System.out.println("Total Error "+totalError);
                    testAccu[0] = test(fileBlock[4], 55);
                    trainAccu[0] = test(fileBlock[0], 55);

                    initWeight();
                    System.out.println("Cross Test 2 of Run " + run);
                    for(int i : trial2)                             
                        train(fileBlock[i]);
                    totalError = 0.5*(target - opOut)*(target - opOut);
                    //System.out.println("Total Error "+totalError);
                    testAccu[1] = test(fileBlock[0], 55);
                    trainAccu[1] = test(fileBlock[1], 55);
                    
                    initWeight();
                    System.out.println("Cross Test 3 of Run " + run);
                    for(int i : trial3)                             
                        train(fileBlock[i]);
                    totalError = 0.5*(target - opOut)*(target - opOut);
                    //System.out.println("Total Error "+totalError);
                    testAccu[2] = test(fileBlock[1], 55);
                    trainAccu[2] = test(fileBlock[2], 55);

                    initWeight();
                    System.out.println("Cross Test 4 of Run " + run);
                    for(int i : trial4)                             
                        train(fileBlock[i]);
                    totalError = 0.5*(target - opOut)*(target - opOut);
                    //System.out.println("Total Error "+totalError);
                    testAccu[3] = test(fileBlock[2], 55);   
                    trainAccu[3] = test(fileBlock[3], 55);

                    initWeight();
                    System.out.println("Cross Test 5 of Run " + run);

                    for(int i : trial5)                             
                        train(fileBlock[i]);
                    totalError = 0.5*(target - opOut)*(target - opOut);
                    //System.out.println("Total Error "+totalError);
                    testAccu[4] = test(fileBlock[3], 55);
                    trainAccu[4] = test(fileBlock[4], 55);

                    for(int j = 0; j < 5; j++) {
                        testSum+= testAccu[j];
                        trainSum+= trainAccu[j];
                    }
                    System.out.println("Average Test Accuracy for Run " + run + ": " + testSum/5.0*100 + "%");
                    System.out.println("Average Train Accuracy for Run " + run + ": " + trainSum/5.0*100 + "%\n");
                }                                                
            }

            //testing /test files
            if(args[0].equalsIgnoreCase("-test")) {
                System.out.println();
                System.out.println("-----Testing Images-----\n");
                int toTrain[] = {0,1,2,3,4};
                initWeight();
                
                for(int i : toTrain) {
                    train(fileBlock[i]);
                }

                File directory = new File(System.getProperty("user.dir")+"/test");
                File[] listOfFiles = directory.listFiles();
                int count = 0;
                for (File file : listOfFiles) {
                    if (file.isFile()) {
                        Scanner scan = new Scanner(file);
                        String row = "";                                 
                        while (scan.hasNextLine()){
                            row += scan.nextLine();
                            row += " ";  
                        }  
                        testBlock[count] = row;
                        count++;            
                    }
                }
                realTest(testBlock, 40);
                
            }
        }
        catch(Exception e)
        {
            System.err.println(e);
            System.err.println("Unknown Error");
            System.exit(5);
        }
    }

    public static void initWeight() {
        //initialize all cells to random(-0.05, 0.05)
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < hiddenLayer; j++) {
                hdWeight[j][i] = (Math.random()*0.1)-0.05;
                opWeight[j] = (Math.random()*0.1)-0.05;
            }
        }
        opBias = (Math.random()*0.1)-0.05;      
    }
    
    //in each block, put 1 female file and put 4 male files - do this 11 times...
    public static void partition() {
        File directory = new File(System.getProperty("user.dir")+"/male");
        File[] MaleFiles = directory.listFiles();
        directory = new File(System.getProperty("user.dir")+"/female");
        File[] FemaleFiles = directory.listFiles();        
        int fcount = 0, mcount = 0;
        int bcount[] = new int[5];
        //0, 5, 10, 15, ..., 45, 50 are female
        for(int i = 0; i < 11; i++) {
            for(int j = 0; j < 5; j++) { //for each block
                //System.out.println(FemaleFiles[fcount].getName());
                //System.out.println(MaleFiles[mcount].getName());
                fileBlock[j][bcount[j]] = fileToString(FemaleFiles[fcount]);
                fcount++;
                if(mcount + 4 > 218)
                    mcount-=2;
                fileBlock[j][bcount[j]+1] = fileToString(MaleFiles[mcount]);
                fileBlock[j][bcount[j]+2] = fileToString(MaleFiles[mcount+1]);
                fileBlock[j][bcount[j]+3] = fileToString(MaleFiles[mcount+2]);
                fileBlock[j][bcount[j]+4] = fileToString(MaleFiles[mcount+3]);
                bcount[j]+=5;
                mcount+=4;
            }
        }
    }

    public static String fileToString(File theFile) {
        String row = ""; 
        try {            
            if(theFile.isFile()) {
                Scanner scan = new Scanner(theFile);
                while (scan.hasNextLine()){
                    row += scan.nextLine();
                    row += " ";
                }            
            }
        }
        catch(Exception e)
        {
            System.err.println("File Not Found");     
        }   
        return row;
    }

    /*reads filestring into pixels*/
    public static void fileToPixels(String filestring) {
        int i = 0;
        String[] splitStr = filestring.split(" ");
        for(String num: splitStr) {
            pixels[i] = Double.parseDouble(num)/255.0;
            //System.out.print(pixels[i][j] + " ");
            i++;
        }
        pixels[i] = 1;  
    }

    /*applies logistic function to net value*/
    public static double squash(double sum) {
        return 1.0/(1.0+Math.exp(1-sum));
    } 

    /*updates net and output values for hidden layer*/
    public static void getHiddenValues() {
        for(int i = 0; i < hiddenLayer; i++) {
            double sum = 0;
            for(int j = 0; j < inputs; j++) {
                sum+= hdWeight[i][j]*pixels[j];
            }
            hdOut[i] = squash(sum);
        }
    }

    /*updates net and output values for output layer*/
    public static void getOutputValues() {
        double sum = 0;
        for(int j = 0; j < hiddenLayer; j++) {
            sum+= opWeight[j]*hdOut[j];
        }
        opOut = squash(sum+opBias);
    }

    public static void updateOpWeight() {
        double dW;
        for(int i = 0; i < hiddenLayer; i++) {
            dW = learningRate*(target - opOut)*opOut*(1.0 - opOut)*hdOut[i];
            tempW[i] = dW;
            //System.out.print(dW + " ");
            //opWeight[i]+= dW;
        }
        opBias+=learningRate*(target - opOut)*opOut*(1.0 - opOut);
        //System.out.println();
    }

    public static void updateHdWeight() {
        double dW;
        for(int i = 0; i < hiddenLayer; i++) {
            for(int j = 0; j < inputs; j++) {
                //dW = learningRate*(target - opOut)*opOut*(1 - opOut)*opWeight[i]*hdOut[i]*(1 - hdOut[i])*compressedImage[j];
                dW = tempW[i]*opWeight[i]*(1.0 - hdOut[i])*pixels[j];
                hdWeight[i][j]+= dW;
            }
            opWeight[i]+=tempW[i];
        }
    }

    /*
    pass in one line and get the average of greyscale
    */
    public static void train(String[] block) {
        for(int a = 0; a < 2; a++) {
            for(int i = 0; i < 55; i++) {
                if(i % 5 == 0)
                    target = targetF;
                else
                    target = targetM;
                fileToPixels(block[i]);
                //compressImage();
                getHiddenValues();   
                getOutputValues();
                updateOpWeight();
                updateHdWeight();
            }  
        }
    }
    
    public static double test(String[] block, int n) {
        double correct = 0;
        for(int i = 0; i < n; i ++) {
            fileToPixels(block[i]);
            //compressImage();        
            getHiddenValues();   
            getOutputValues();
            //System.out.print(i + " " + opOut + " ");
            if(opOut > 0.62555) {
                if(i % 5 != 0)
                    correct++;
                //System.out.println("Male");
            }
            else {
                if(i % 5 == 0) 
                    correct++;
                //System.out.println("Female");
            }
        }  
        double acc = correct/55.0;
        System.out.println("Accruracy = " + acc*100 + "%"); 
        return acc;
    }

    public static void realTest(String[] block, int n) {
        double correct = 0, conf;
        for(int i = 0; i < n; i ++) {
            fileToPixels(block[i]);
            //compressImage();        
            getHiddenValues();   
            getOutputValues();
            //System.out.print(i + " " + opOut + " ");
            if(opOut > 0.62555) {
                conf = 1 - Math.abs(opOut-targetM)/targetM;
                System.out.println(" Male " + conf*100 + "% Confidence");
            }
            else {
                conf = 1 - targetF/Math.abs(opOut-targetF);
                System.out.println(" Female " + conf*100 + "% Confidence");
            }
        }  
    }
    
    public static void saveWeightTable() throws FileNotFoundException, UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter("weight1.txt", "UTF-8");
        int count = 1;
        for (int i = 0; i < inputs; i++){
            for (int j = 0; j < hiddenLayer; j++){
                writer.print(hdWeight[i][j] + " ");
                count += 1;
                if (count == 5){
                    writer.print("\n");
                    count = 1;
                }
            }
        }
        writer.close();
        
        PrintWriter newWriter = new PrintWriter("weight2.txt", "UTF-8");
        for (int i = 1; i <= hiddenLayer; i++){
            newWriter.print(opWeight[i] + " ");
        }
        newWriter.close();
    }
}

