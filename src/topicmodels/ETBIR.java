package topicmodels;

import java.io.*;
import java.lang.reflect.Array;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import Analyzer.DocAnalyzer;
import json.JSONArray;
import json.JSONObject;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import structures.*;
import utils.Utils;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class ETBIR{

    protected int m_varMaxIter;
    protected double m_varConverge;
    protected int m_emMaxIter;
    protected double m_emConverge;

    protected int vocabulary_size;
    protected int number_of_topics;
    protected int number_of_users;
    protected int number_of_items;

    public _User[] us;
    public _Item[] is;
    public ArrayList<_Review> corpus;

    protected double m_rho;
    protected double m_sigma;
    protected double[] m_alpha;
    protected double[][] m_beta; //topic_term_probability

    public ETBIR(int emMaxIter, double emConverge,  int varMaxIter, double varConverge, //stop criterion
                           int nTopics, int nVocab //user pre-defined arguments
                           ) {
        this.m_emMaxIter = emMaxIter;
        this.m_emConverge = emConverge;
        this.m_varMaxIter = varMaxIter;
        this.m_varConverge = varConverge;

        this.number_of_topics = nTopics;
        this.vocabulary_size = nVocab;
    }

    protected void initModel(){
        System.out.println("------------ initializing model ------------");

        this.m_alpha = new double[number_of_topics];
        this.m_beta = new double[number_of_topics][vocabulary_size];


    }

    public double calc_E_step(_Review d, _User u, _Item i) {
        double last = 0.0;
        if (m_varConverge > 0) {
            last = calc_log_likelihood(d, us, is);
        }

        double current = last, converge = 0.0;
        int iter = 0;

        do {
            update_phi(d);

            update_zeta(d);
            update_mu(d, u ,i);
            update_SigmaTheta(d);

            update_nu(u,is);
            update_SigmaP(u, is, d);

            update_eta(i, us, d);

            if (m_varConverge > 0) {
                current = calc_log_likelihood(d, us, is);
                converge = Math.abs((current - last) / last);
                last = current;
                if (converge < m_varConverge) {
                    break;
                }
            }
        } while (++iter < m_varMaxIter);
    }

    //variational inference for p(z|w,\phi) for each document
    public void update_phi(_Review d){
        double logSum, v;
        int wid;
        _SparseFeature[] fv = d.getSparse();

        for (int n = 0; n < fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for (int k = 0; k < number_of_topics; k++) {
                d.m_phi[n][k] = Math.log(m_beta[k][wid]) + d.m_mu[k];
            }
            // normalize
            logSum = Utils.logSum(d.m_phi[n]);
            for (int k = 0; k < number_of_topics; k++) {
                d.m_phi[n][k] = Math.exp(d.m_phi[n][k] - logSum);
            }
        }
    }

    //variational inference for p(\theta|\mu,\Sigma) for each document
    public void update_zeta(_Review d){
        //estimate zeta
        d.m_zeta = 0;
        for (int k = 0; k < number_of_topics; k++) {
            d.m_zeta += Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
        }

    }

    //variational inference for p(\theta|\mu,\Sigma) for each document: Quasi-Newton, LBFGS(minimize)
    public void update_mu(_Review d, _User u, _Item i){
        int[] iflag = {0}, iprint = {-1,3};
        double fValue = 0.0;
        double[] m_muG = new double[number_of_topics]; // gradient for mu
        double[] mu_diag = new double[number_of_topics];
        int N = d.getDocInferLength();

        Arrays.fill(mu_diag, 0.0);

        double moment, zeta_stat = 1.0 / d.m_zeta;
        try{
            do {
                //update gradient of mu
                for (int k = 0; k < number_of_topics; k++) {
                    moment = Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
                    m_muG[k] = -(-m_rho * (d.m_mu[k] - Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                            + Utils.sumOfColumn(d.m_phi, k) - N * zeta_stat * moment);//-1 because LBFGS is minimization
                    fValue += -0.5 * m_rho * (d.m_mu[k] * d.m_mu[k]
                            - 2 * d.m_mu[k] * Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                            + d.m_mu[k] * Utils.sumOfColumn(d.m_phi, k) - N * zeta_stat * moment;
                }
                fValue = -fValue; //LBFGS is about minimization
                LBFGS.lbfgs(number_of_topics,4, d.m_mu, fValue, m_muG,false, mu_diag, iprint, 1e-6, 1e-32, iflag);
            } while (iflag[0] != 0);
        }catch (ExceptionWithIflag e){
            e.printStackTrace();
        }
    }

    //variational inference for p(\theta|\mu,\Sigma) for each document: Quasi-Newton, LBFGS
    public void update_SigmaTheta(_Review d){
        int[] iflag = {0}, iprint = {-1,3};
        double fValue = 0.0;
        int N = d.getDocInferLength();
        protected double[] m_SigmaG = new double[number_of_topics]; // gradient for Sigma
        double[] sigma_diag = new double[number_of_topics];
        Arrays.fill(sigma_diag, 0.0);

//        double[] log_sigma = new double[number_of_topics];
//        for(int k=0; k < number_of_topics; k++){
//            log_sigma[k] = Math.log(d.m_Sigma[k]);
//        }

        double moment, zeta_stat = 1.0 / d.m_zeta;
        try{
            do {
                //update gradient of sigma
                for (int k = 0; k < number_of_topics; k++) {
                    moment = Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
                    m_SigmaG[k] = -0.5 * (1.0 / d.m_Sigma[k] - m_rho - N * zeta_stat * moment); //-1 because LBFGS is minimization
                    fValue += -0.5 * m_rho * d.m_Sigma[k] - N * zeta_stat * moment + 0.5 * Math.log(d.m_Sigma[k]);
                }
                fValue = -fValue;
                LBFGS.lbfgs(number_of_topics,4, d.m_Sigma, fValue, m_SigmaG,false, sigma_diag, iprint, 1e-6, 1e-32, iflag);

            } while(iflag[0] != 0);
        }catch (ExceptionWithIflag e){
            e.printStackTrace();
        }
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    public void update_nu(_User u, _Item[] is){
        RealMatrix eta_stat_sigma = MatrixUtils.createRealIdentityMatrix(number_of_topics).scalarMultiply(m_sigma);
        for (int item_i = 0; item_i < number_of_items; item_i++) {
            RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(is[item_i].m_eta);
            double eta_0 = Utils.sumOfArray(is[item_i].m_eta);
            RealMatrix eta_stat_i = MatrixUtils.createRealDiagonalMatrix(is[item_i].m_eta).add(eta_vec.multiply(eta_vec.transpose()));
            eta_stat_sigma.add(eta_stat_i.scalarMultiply(m_rho / (eta_0 * (eta_0 + 1.0))));
        }
        eta_stat_sigma = new LUDecomposition(eta_stat_sigma).getSolver().getInverse();
        for (int k = 0; k < number_of_topics; k++) {
            u.m_SigmaP[k] = eta_stat_sigma.getData();
        }
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    public void update_SigmaP(_User u, _Item[] is, _Review d){
        RealMatrix eta_stat_sigma = MatrixUtils.createRealMatrix(u.m_SigmaP[0]);
        RealMatrix eta_stat_nu = MatrixUtils.createColumnRealMatrix(new double[number_of_topics]);

        for (int k = 0; k < number_of_topics; k++) {
            for (int item_i = 0; item_i < number_of_items; item_i++) {
                RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(is[item_i].m_eta);
                double eta_0 = Utils.sumOfArray(is[item_i].m_eta);
                eta_stat_nu.add(eta_vec.scalarMultiply(d.m_mu[k] / eta_0));
            }
            u.m_nuP[k] = eta_stat_sigma.multiply(eta_stat_nu).scalarMultiply(m_rho).getColumn(0);
        }
    }

    public void update_eta(_Item i, _User[] us, _Review d){
        int[] iflag = {0}, iprint = {-1,3};
        double fValue = 0.0;
        double[] m_etaG = new double[number_of_topics];
        double[] eta_diag = new double[number_of_topics];
        Arrays.fill(eta_diag, 0.0);

        double eta_0 = Utils.sumOfArray(i.m_eta);
        try{
            do{
                for(int k = 0; k < number_of_topics; k++){

                    //might be optimized using global stats
                    double gTerm1 = 0.0;
                    double gTerm2 = 0.0;
                    double gTerm3 = 0.0;
                    double gTerm4 = 0.0;
                    double term1 = 0.0;
                    double term2 = 0.0;
                    for(int uid = 0; uid < number_of_users; uid++){
                        for(int j = 0; j < number_of_topics; j++){
                            gTerm1 += us[uid].m_nuP[j][k] * d.m_mu[j];
                            term1 += i.m_eta[k] * us[uid].m_nuP[j][k] * d.m_mu[j];

                            for(int l = 0;l < number_of_topics; l ++){
                                gTerm2 += i.m_eta[l] * us[uid].m_nuP[j][l] * d.m_mu[j];

                                gTerm3 += i.m_eta[l] * (us[uid].m_SigmaP[j][l][k] + us[uid].m_nuP[j][l] * us[uid].m_nuP[j][k]);
                                if(l == k){
                                    gTerm3 += (i.m_eta[l] + 1.0) * (us[uid].m_SigmaP[j][l][k] + us[uid].m_nuP[j][l] * us[uid].m_nuP[j][k]);
                                }

                                term2 += i.m_eta[l] * i.m_eta[k] * (us[uid].m_SigmaP[j][l][k] + us[uid].m_nuP[j][l] * us[uid].m_nuP[j][k]);
                                if(l == k){
                                    term2 += i.m_eta[k] * (us[uid].m_SigmaP[j][k][k] + us[uid].m_nuP[j][k] * us[uid].m_nuP[j][k]);
                                }

                                for(int p = 0; p < number_of_topics; p++){
                                    gTerm4 += i.m_eta[l] * i.m_eta[p] * (us[uid].m_SigmaP[j][l][p] + us[uid].m_nuP[j][l] * us[uid].m_nuP[j][p]);
                                    if(p == l){
                                        gTerm4 += i.m_eta[p] * (us[uid].m_SigmaP[j][l][p] + us[uid].m_nuP[j][l] * us[uid].m_nuP[j][p]);
                                    }
                                }
                            }
                        }
                    }

                    m_etaG[k] = -(Utils.trigamma(i.m_eta[k]) * (m_alpha[k] - i.m_eta[k])
                            - Utils.trigamma(eta_0) * (Utils.sumOfArray(m_alpha) - eta_0)
                            + m_rho * gTerm1 / eta_0 - m_rho * gTerm2 / (Math.pow(eta_0, 2))
                            - m_rho * gTerm3 / (2 * eta_0 * (eta_0 + 1.0))
                            + m_rho * (2 * eta_0 + 1.0) * gTerm4 / (2 * Math.pow(eta_0, 2) * Math.pow(eta_0 + 1.0, 2)));

                    fValue += (m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                            - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0));
                }
                fValue = -fValue;
                LBFGS.lbfgs(number_of_topics,4, i.m_eta, fValue, m_etaG,false, eta_diag, iprint, 1e-6, 1e-32, iflag);
            }while(iflag[0] != 0);
        }catch (ExceptionWithIflag e){
            e.printStackTrace();
        }
    }

    public void calc_M_step(int iter) {
        //maximize likelihood for \rho of p(\theta|P\gamma, \rho)
        double term1 = 0.0;

        for(int item_i=0; item_i < number_of_items; item_i++){
            for(int user_u=0; user_u < number_of_users; user_u++){

            }
        }

    }

    protected double calc_log_likelihood(_Review d, _User[] us, _Item[] is) {

    }

    protected void EM(){

    }

    public void processData(String fileName){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
            StringBuffer buffer = new StringBuffer(1024);
            String line;

            while((line=reader.readLine())!=null) {
                buffer.append(line);
            }
            reader.close();
            JSONArray jarry = new JSONArray(buffer.toString());
        } catch (Exception e) {
            System.out.print("! FAIL to load json file...");
        }


    }

    public void readData(String fileName){

    }

    public void readVocabulary(String fileName){

    }

    public static void main(String[] args) throws FileNotFoundException{
        String outFileName = "output_ETBIR.txt";
        PrintStream out = new PrintStream(new FileOutputStream(outFileName));

        String dataFileName = "../myData/review.json";
        String vocFileName = "input/vocab.dat";

        int topic_number = 30;
        int vocab_size = 0;

        int varMaxIter = 20;
        double varConverge = 1e-6;

        int emMaxIter = 1000;
        double emConverge = 1e-3;

        ETBIR etbirModel = new ETBIR(emMaxIter, emConverge, varMaxIter, varConverge, topic_number, vocab_size);
        etbirModel.processData(dataFileName);
//        etbirModel.readData(dataFileName);
//        etbirModel.readVocabulary(vocFileName);
//        etbirModel.EM();
    }
}
