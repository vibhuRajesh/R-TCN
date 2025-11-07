import java.util.ArrayList;
import java.util.Arrays;
import java.io.Serializable;

public class Network implements Serializable{


    //treat like Linked List:
    private int length ;

    //keep track of size
    private int numConv = 0;
    private int numFCN;
    private FCN firstFCN;

    //"head" of list
    private Layer first;
    //while building network, last could be Convolution or FCN
    private Layer last;

    private ArrayList<FCN> fcnLayers;

    //ground truth
    private int[] answer;

    //also initialized in this.train() based on number of epochs passed in for training
    private double[] trainLoss;


    private double[] trainAcc;

    //confusion matrix for precision, recall and f1
    //conf[][0] = TruePos, conf[][1] = TrueNeg, conf[][2] = FalsePos, conf[][3] = FalseNeg          ([] = epoch num)
    private double[][] conf;

    private double[] precision;
    private double[] recall;
    private double[] f1;

    //folds[][][] => one window is 1 feature x w x p
    double[][][] folds;
    //folds[][][][] => folds[i] = n features x w x p
    double[][][][] Folds;

    private double testLoss;
    private double testAcc;

    //CONSTRUCTORS
    //1) Empty network
    //2) Premade layer passed in
    //3) Manually instantiate first layer

    /**
     * default constructor for empty network
     */
    public Network(){
        this.length = 0;
    }

    /**
     * overloaded constructor for initializing with head
     * @param first: Convolution layer to set as first
     */
    public Network(Convolution first, int s, int st){
        this.first = first;
        this.last = this.first;
        this.length++;
        this.numConv++;
        //keeps track for reinstantiating with proper window size in validation
        fcnLayers = new ArrayList<>();
    }

    /**
     * Manually instantiate the first layer
     * @param f: input split into folds for layer (output from DataReader is 2D, 3D if multichannel)
     * @param s: window start
     * @param st: window stop
     * @param n: number of filters to apply
     * @param k: kernel size (size of each filter)
     * @param d: dilation factor
     * @param r: if layer takes in a residual input from previous layer (false, just needed in Convolution)
     * @param re: residual to be added (null, same as above)
     * @param l: learning rate
     * @param a: answer array (since first layer)
     * @param i: which fold to use
     */
    public Network( double[][][] f, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[][] a, int i){

        this.folds = f;
        this.answer = a[i];
        this.first = new Convolution(f[i], s, st, n, k, d, r, re, l, a[i]);
        this.last = this.first;
        this.length++;
        this.numConv++;

    }

    public Network( double[][][][] f, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[][] a, int i){

        this.Folds = f;
        this.answer = a[i];
        this.first = new Convolution(f[i], s, st, n, k, d, r, re, l, a[i]);
        this.last = this.first;
        this.length++;
        this.numConv++;

    }

    public Network( double[][][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[] a ){
        this.answer = a;
        this.first = new Convolution(i, s, st, n, k, d, r, re, l, a);
        this.last = this.first;
        this.length++;
        this.numConv++;

    }




    //Adding layers: have option to do manually with overloaded methods, or with previously instantiated layer passed in

    //each of the constructors have method, one of Convolution's constructor's is Network constructor
        //Convolution:
            //3D input: standard Conv layer
        //FCN
            //FCN: simply passed in
            //3D input: first after convolution
            //2D input (all following after 3D)

    public void addConv(Convolution toAdd){

        //empty network constructor used
        if(this.length == 0){
            this.first = toAdd;
        }

        else if(this.length == 1){
            //Layer.connect will assign toAdd's .isLast() to true
            Layer.connect(this.first, toAdd);
            //Makes sure value matches in Network
        }
        else{
            Layer.connect(this.last, toAdd);
        }

        //increment length
        this.last = toAdd;
        this.length++;
        this.numConv++;
    }

    //typically used for non-residual layers (can be used if Convolution re is pre instantiated
    public void addConv(double[][][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l){
        Convolution toAdd = new Convolution(i, s, st, n, k, d, r, re, l);
        this.addConv(toAdd);
    }

    /**
     * more used for residual layers (only called when residual needed)
     * @param re: index of layer to take residual input from
     */
    public void addConv(double[][][] i, int s, int st, int n, int k, int d, boolean r, int re, double l){
        Convolution res;
        if(r){
            res = (Convolution) (this.get(re));
        }
        else{
            res = null;
        }

        Convolution toAdd = new Convolution(i, s, st, n, k, d, r, res, l);
        this.addConv(toAdd);
    }



    /**
     * PreInstantiated FCN layer passed in
     * @param toAdd: FCN layer to add
     */
    public void addFCN(FCN toAdd){
        if(this.length == 0){
            this.first = toAdd;
        }

        else if(this.length == 1){
            //Layer.connect will assign toAdd's .isLast() to true
            Layer.connect(this.first, toAdd);
            //Makes sure value matches in Network
        }
        else{
            Layer.connect(this.last, toAdd);
        }

        this.last = toAdd;
        this.length++;
        this.numFCN++;

    }

    /**
     * 2D FCN Layer to be added
     * @param i: 2D input array
     * @param o: output neurons (num neurons)
     * @param l: learning rate
     */
    public void addFCN(double[][] i, int o, double l){
        FCN toAdd = new FCN(i, o, l);
        this.addFCN(toAdd);
    }

    /**
     * 3D FCN Layer to be added
     * @param s: start idx
     * @param st: stop idx
     */
    public void addFCN(double[][][] i, int o, double l, int s, int st){
        FCN toAdd = new FCN(i, o, l, s, st);
        this.addFCN(toAdd);
        this.firstFCN = toAdd;
    }
    public static void saveNetwork(Network n, String f){

    }


    /**
     * Training method, calculates accuracy, precision, recall and f1, saves state based on highest f1 score
     * @param epochs: number of epochs to train for
     * @param folder: folder to save each set of weight/bias/ tensors
     * @param save: whether to save file or not
     *
     *
     *
     */
    public void train(int epochs, String folder, boolean save){


        //epoch with highest Accuracy
        double maxAcc = -1;
        Network max = new Network();

        //initialize ArrayList to add Layer info (all based on epoch size)
        //keeps track of model state, initialized within .train()
        //weights


        //keep track of loss/acc/f1 by epoch
        this.trainLoss = new double[epochs];
        this.trainAcc = new double[epochs];

        this.conf = new double[epochs][4];
        this.precision = new double[epochs];
        this.recall = new double[epochs];
        this.f1 = new double[epochs];


        //number of epochs
        for(int i = 0; i < epochs; i++){
            System.out.println("START EPOCH: " + i);


            //forward pass node
            Layer dummy = this.first;

            //make sure  dummy.getNext is != null instead of dummy != null:
                // if dummy == null when exiting loop, null cannot reference .backward() (similar for .backward loop)
            while(dummy.getNext() != null){

                //individual layer step forward
                dummy.forward();
                //move to next layer
                dummy = dummy.getNext();

            }
            dummy.forward();


            //plain accuracy and loss
            this.trainAcc[i] = ( (FCN) dummy).accuracy;
            this.trainLoss[i] = ( (FCN) dummy).loss();

            //maintain reference of maximum accuracy
                        //for precision, recall and f1 score
            double[] guess = ( (FCN) dummy).guess();


            //backward pass (no information needed to store)
            while(dummy.getPrev() != null){

                dummy.backward();
                dummy = dummy.getPrev();

            }
            //same thing as forward: backward pass of first layer
            dummy.backward();


            System.out.println("EPOCH: " + i);
            System.out.println("GUESS: " + Arrays.toString(guess));
            System.out.println("ACC" + this.trainAcc[i]);
            System.out.println("LOSS" + this.trainLoss[i]);
            System.out.println("ALL ACC " + Arrays.toString(this.trainAcc));
            System.out.println();
            System.out.println("LOSS" + Arrays.toString(this.trainLoss));
//            System.out.println("F1" + Arrays.toString(this.f1));

            if(this.trainAcc[i] >= maxAcc){
                maxAcc = this.trainAcc[i];
                max = this;
                System.out.println("MAX " + maxAcc);
                System.out.println("MAX EPOCH " + i);

            }
            System.out.println("\nEND EPOCH: " + i+ "\n--------------------");
        }

        if(save){
            Save.saveNetwork(max, folder);
        }
    }


    /**
     * returns single accuracy value (of epoch)
     * @param guess: predicted value
     * @param truth: ground truth to compare to
     * @return accuracy value
     */
    public double accuracy(double[] guess, int[] truth){
        double acc;

        double count = 0;
        double total = guess.length;

        //guess.length is the window length
        for(int i = 0; i < guess.length; i++){

            //percent that guessed correct class
            if(guess[i] == truth[i]){
                count++;
            }

        }
        return (count / total);
    }


    /**
     * 0-indexed
     * @param index
     * @return Layer at particular index
     */
    public Layer get(int index){
        if(index < this.length && index >= 0){

            Layer temp = this.first;
            while(temp.getIndex() != index){
                temp = temp.getNext();
            }

            return temp;

        }
        else{
            return null;
        }
    }


    public static Network copy(Convolution first, int s, int st){
        Layer iter = first;
        Network copy = new Network(first.getInput(), s, st, first.getNumFilters(), first.getKernel(), first.getDilation(), first.isResid(), null, first.getLearnRate(), first.getAnswer());
        while(iter.getNext() != null){
            iter = iter.getNext();
            //Convolution half of network
            if(iter instanceof Convolution){

                Convolution temp = (Convolution) iter;
                //if not residual layer, then takes in no layer or index as input
                if(!temp.isResid()) {
                    copy.addConv(((Convolution) copy.getLast()).getOutput(), s, st, temp.getNumFilters(), temp.getKernel(), temp.getDilation(), temp.isResid(), null, temp.getLearnRate());
                }

                else{
                    Convolution r = temp.getResid();
                    int idx = r.getIndex();
                    Convolution res = (Convolution) copy.get(idx);
                    copy.addConv(((Convolution) copy.getLast()).getOutput(), s, st, temp.getNumFilters(), temp.getKernel(), temp.getDilation(), temp.isResid(), res, temp.getLearnRate());
                    System.out.println();

                }

            }


            else{
                FCN temp = (FCN) iter;

                if(iter.getPrev() instanceof Convolution){
                    copy.addFCN(((Convolution) copy.getLast()).getOutput(), temp.getNumFilters(), temp.getLearnRate(), s, st);
                }

                else{
                    copy.addFCN(((FCN) copy.getLast()).getOutput(), temp.getNumFilters(), temp.getLearnRate());
                }

            }

        }


        return copy;
    }



    //checker method that isLast() property that Layers have matches Network's last

    public boolean checkLast(){
        return this.last.isLast();
    }
    public Layer getFirst(){
        return this.first;
    }

    public void setRho(double rho){
        Layer iter = this.first;
        while(iter != null){

            if(iter instanceof Convolution){
                ((Convolution) iter).setRho(rho);
            }
            else{
                ((FCN) iter).setRho(rho);
            }
            iter = iter.getNext();


        }
    }



    public Layer getLast(){
        return this.last;
    }
    public int getLength(){
        return this.length;
    }

    public double[] getTrainAcc() {
        return trainAcc;
    }

}
