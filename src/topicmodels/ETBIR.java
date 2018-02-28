package topicmodels;

import java.io.*;
import java.lang.reflect.Array;
import java.text.ParseException;
import java.util.*;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import structures.*;
import utils.Utils;

import javax.rmi.CORBA.Util;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class ETBIR{

    protected int m_varMaxIter;
    protected double m_varConverge;
    protected int m_emMaxIter;
    protected double m_emConverge;

    public int vocabulary_size;
    public int number_of_topics;
    public int number_of_users;
    public int number_of_items;

    public _User[] m_users;
    public HashMap<String, Integer> m_usersIndex;
    public _Product[] m_items;
    public HashMap<String, Integer> m_itemsIndex;
    public _Corpus m_corpus;
    public HashMap<String, Integer> m_reviewIndex;

    public double m_rho;
    public double m_sigma;
    public double[][] m_beta; //topic_term_probability
    public double[] m_alpha;
    public double[] m_alphaG;
    public double[] m_alphaH;

    public double dAlpha;
    public double dSigma;
    public double dRho;
    public double dBeta;


    public double[] m_eta0;
    public double[] m_etaStats;
    public double[][] m_word_topic_stats;
    public double m_pStats;
    public double m_thetaStats;
    public double[] m_eta_p_Stats;
    public double[] m_eta_mean_Stats;

    public ETBIR(int emMaxIter, double emConverge,  int varMaxIter, double varConverge, //stop criterion
                           int nTopics, //user pre-defined arguments
                           double dalpha, double dsigma, double drho, double dbeta) {
        this.m_emMaxIter = emMaxIter;
        this.m_emConverge = emConverge;
        this.m_varMaxIter = varMaxIter;
        this.m_varConverge = varConverge;

        this.number_of_topics = nTopics;

        this.dAlpha = dalpha;
        this.dSigma = dsigma;
        this.dRho = drho;
        this.dBeta = dbeta;
    }

    public void loadCorpus(_Corpus corpus){
        System.out.println("Loading data to model...");

        this.m_corpus = corpus;
        m_usersIndex = new HashMap<String, Integer>();
        m_itemsIndex = new HashMap<String, Integer>();
        m_reviewIndex = new HashMap<String, Integer>();

        int u_index = 0, i_index = 0;
        for(_Doc d : m_corpus.getCollection()){
            String userID = d.getTitle();
            String itemID = d.getItemID();

            if(!m_usersIndex.containsKey(userID)){
                m_usersIndex.put(userID, u_index++);
            }

            if(!m_itemsIndex.containsKey(itemID)){
                m_itemsIndex.put(itemID, i_index++);
            }
        }
        m_users = new _User[m_usersIndex.size()];
        for(Map.Entry<String, Integer> entry: m_usersIndex.entrySet()){
            m_users[entry.getValue()] = new _User(entry.getKey());
        }

        m_items = new _Product[m_itemsIndex.size()];
        for(Map.Entry<String, Integer> entry: m_itemsIndex.entrySet()){
            m_items[entry.getValue()] = new _Product(entry.getKey());
        }

        this.number_of_items = m_items.length;
        this.number_of_users = m_users.length;
        this.vocabulary_size = m_corpus.getFeatureSize();

        for(int d = 0; d < m_corpus.getCollection().size(); d++){
            _Doc doc = m_corpus.getCollection().get(d);
            int uIndex = m_usersIndex.get(doc.getTitle());
            int iIndex = m_itemsIndex.get(doc.getItemID());
            m_reviewIndex.put(iIndex + "_" + uIndex, d);
        }

        System.out.println("-- vocabulary size: " + vocabulary_size);
        System.out.println("-- corpus size: " + m_reviewIndex.size());
        System.out.println("-- item number: " + number_of_items);
        System.out.println("-- user number: " + number_of_users);
    }

    //create space; initial parameters
    public void initModel(){
        this.m_alpha = new double[number_of_topics];
        this.m_beta = new double[number_of_topics][vocabulary_size];
        this.m_alphaG = new double[number_of_topics];
        this.m_alphaH = new double[number_of_topics];

        //initialize parameters
        Random r = new Random();
        m_rho = dRho;
        m_sigma = dSigma;
        Arrays.fill(m_alpha, dAlpha);
        double val = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            double sum = 0.0;
            for(int v = 0; v < vocabulary_size; v++){
                val = r.nextDouble() + dBeta;
                sum += val;
                m_beta[k][v] = val;
            }

            for(int v = 0; v < vocabulary_size; v++){
                m_beta[k][v] = m_beta[k][v] / sum;
            }
        }

        this.m_eta0 = new double[number_of_items];
        this.m_etaStats = new double[number_of_topics];
        this.m_word_topic_stats = new double[number_of_topics][vocabulary_size];
        this.m_eta_p_Stats = new double[number_of_items];
        this.m_eta_mean_Stats = new double[number_of_items];
    }

    public void initDoc(_Doc doc){
        doc.m_zeta = 1.0;
        doc.m_mu = new double[number_of_topics];
        doc.m_Sigma = new double[number_of_topics];
        doc.m_phi = new double[doc.getSparse().length][number_of_topics];
        Arrays.fill(doc.m_mu, 0.5);
        Arrays.fill(doc.m_Sigma, 1.0);
        for(int i=0;i < doc.getSparse().length;i++){
            Arrays.fill(doc.m_phi[i], 1.0/number_of_topics);
        }
    }

    public void initUser(_User user){

        user.m_nuP = new double[number_of_topics][number_of_topics];
        user.m_SigmaP = new double[number_of_topics][number_of_topics][number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            Arrays.fill(user.m_nuP[k], 1.0);
            for(int l = 0; l < number_of_topics; l++){
                Arrays.fill(user.m_SigmaP[k][l], 0.1);
                user.m_SigmaP[k][l][l] = 0.1;
            }
        }

    }

    public void initItem(_Product item){

        item.m_eta = new double[number_of_topics];
        Arrays.fill(item.m_eta, dAlpha);

    }

    protected void initStats(){
        Arrays.fill(m_eta0, 0.0);
        Arrays.fill(m_etaStats, 0.0);
        for(int k = 0; k < number_of_topics; k++){
            Arrays.fill(m_word_topic_stats[k], 0);
        }
        m_pStats = 0.0;
        m_thetaStats = 0.0;
        Arrays.fill(m_eta_p_Stats, 0.0);
        Arrays.fill(m_eta_mean_Stats, 0.0);
    }

    protected void updateStats(_Doc review){
        // update m_eta0 for most parameters regarding eta
        for(int i = 0; i < number_of_items; i++){
            m_eta0[i] = Utils.sumOfArray(m_items[i].m_eta);
        }

        // update m_etaStats for updating alpha
        for(int i = 0; i < number_of_items; i++){
            for(int k = 0; k < number_of_topics;k++){
                m_etaStats[k] += Utils.digamma(m_items[i].m_eta[k]) - Utils.digamma(m_eta0[i]);
            }
        }

        // update m_word_topic_stats for updating beta
        _SparseFeature[] fv = review.getSparse();
        for(int d = 0; d < number_of_items; d++){
            for(int k = 0; k < number_of_topics; k++){
                for(int n = 0; n < fv.length; n++){
                    int wid = fv[n].getIndex();
                    double v = fv[n].getValue();
                    m_word_topic_stats[k][n] += v * review.m_phi[n][k];
                }
            }
        }

        // update m_pStates for updating sigma
        for(int u = 0; u < number_of_users; u++){
            for(int k = 0; k < number_of_topics; k++){
                for(int l = 0; l < number_of_topics; l++){
                    m_pStats += m_users[u].m_SigmaP[k][l][l] + m_users[u].m_nuP[k][l] * m_users[u].m_nuP[k][l];
                }
            }
        }

        // update m_thetaStats for updating rho
        for(int d = 0; d < m_corpus.getSize(); d++){
            _Doc doc = m_corpus.getCollection().get(d);
            for(int k = 0; k < number_of_topics; k++){
                m_thetaStats += doc.m_Sigma[k] + doc.m_mu[k] * doc.m_mu[k];
            }
        }

        // update m_eta_p_stats for updating rho and eta
        // update m_eta_mean_stats for updating rho and eta
        for(int i = 0; i < number_of_items; i++) {
            for (int u = 0; u < number_of_users; u++) {
                String dKey = i + "_" + u;
                _Doc currentD = m_corpus.getCollection().get(m_reviewIndex.get(dKey));
                for (int k = 0; k < number_of_topics; k++) {
                    for (int l = 0; l < number_of_topics; l++) {
                        m_eta_mean_Stats[i] += m_items[i].m_eta[l] * m_users[u].m_nuP[l][k] * currentD.m_mu[k];

                        for (int j = 0; j < number_of_topics; j++) {
                            double term1 = m_users[u].m_SigmaP[k][l][j] + m_users[u].m_nuP[k][l] * m_users[u].m_nuP[k][j];
                            m_eta_p_Stats[i] += m_items[i].m_eta[l] * m_items[i].m_eta[j] * term1;
                            if (j == l) {
                                term1 = m_users[u].m_SigmaP[k][l][j] + m_users[u].m_nuP[k][l] * m_users[u].m_nuP[k][j];
                                m_eta_p_Stats[i] += m_items[i].m_eta[l] * term1;
                            }
                        }
                    }
                }
            }
            m_eta_mean_Stats[i] /= m_eta0[i];
            m_eta_p_Stats[i] /= m_eta0[i] * (m_eta0[i] + 1.0);
        }
    }

    protected double E_step(){

        double totalLikelihood = 0.0;

        int i = 0;
        for(_Doc d : m_corpus.getCollection()){
            System.out.println("***************** doc " + i + " ****************");
            String userID = d.getTitle();
            String itemID = d.getItemID();
            _User currentU = m_users[m_usersIndex.get(userID)];
            _Product currentI = m_items[m_itemsIndex.get(itemID)];

            double cur = varInference(d, currentU, currentI);
            if (Double.isNaN(cur)){
                System.out.println("cur is NaN!!!!!!!!!!!!!!!!!!!!!!!!!");
                continue;
            }
            totalLikelihood += cur;
            System.out.println("currentVarInference: " + cur + "; totalLikelihood: " + totalLikelihood);
            updateStats(d);
            i++;
        }

        return totalLikelihood;
    }

    protected double varInference(_Doc d, _User u, _Product i) {
        double last = 0.0;
        if (m_varConverge > 0) {
            last = calc_log_likelihood(d, u, i);
        }

        double current = last, converge = 0.0;
        int iter = 0;

        do {
            update_SigmaP(u);
            update_nu(u, d);

//            update_eta(i, d);

            update_zeta(d);
            update_mu(d, u ,i);
            update_zeta(d);
            update_SigmaTheta(d);
            update_zeta(d);

            update_phi(d);

            if (m_varConverge > 0) {
                current = calc_log_likelihood(d, u, i);
                converge = Math.abs((current - last) / last);
                last = current;
                if (converge < m_varConverge) {
                    break;
                }
            }
        } while (++iter < m_varMaxIter);

        double sumPhi = 0.0;
        for(int k=0;k < d.m_phi.length;k++){
            sumPhi += Utils.sumOfArray(d.m_phi[k]);
        }
        System.out.println("d.phi: " + sumPhi + "; d.zeta: " + d.m_zeta
                + "; d.mu: " + Utils.sumOfArray(d.m_mu) + "; d.Sigma: " + Utils.sumOfArray(d.m_Sigma));

        double sumNu = 0.0;
        for(int k= 0; k < number_of_topics;k++){
            sumNu += Utils.sumOfArray(u.m_nuP[k]);
        }
        RealMatrix sigmaTemp = MatrixUtils.createRealMatrix(u.m_SigmaP[0]);
        double sigmaDeter = new LUDecomposition(sigmaTemp).getDeterminant();
        System.out.println("u.nu: " + sumNu + "; u.Sigma: " + sigmaDeter);

        double sumBeta = 0.0;
        for(int k= 0; k < number_of_topics;k++){
            for(int v = 0; v < vocabulary_size;v++){
                sumBeta += Math.log(m_beta[k][v]);
            }
        }
        System.out.println("m_beta: " + sumBeta);
        return current;
    }

    //variational inference for p(z|w,\phi) for each document
    public void update_phi(_Doc d){
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
    public void update_zeta(_Doc d){
        //estimate zeta
        d.m_zeta = 0;
        for (int k = 0; k < number_of_topics; k++) {
            d.m_zeta += Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
        }

    }

    //variational inference for p(\theta|\mu,\Sigma) for each document: Quasi-Newton, LBFGS(minimize)
    public void update_mu1(_Doc d, _User u, _Product i){
        int[] iflag = {1}, iprint = {-1,3};
        double fValue;
        double[] m_mu = new double[number_of_topics];
        double[] m_muG = new double[number_of_topics]; // gradient for mu
        double[] mu_diag = new double[number_of_topics];
        int N = d.getSparse().length;

        Arrays.fill(mu_diag, 0.0);
        for(int k = 0; k < number_of_topics; k++){
            m_mu[k] = d.m_mu[k];
        }

        double moment, zeta_stat = 1.0 / d.m_zeta;
        int iter = 0, iterMax = 10;
        try {
            do {
                //update gradient of mu
                fValue = 0.0;
                for (int k = 0; k < number_of_topics; k++) {
                    moment = Math.exp(m_mu[k] + 0.5 * d.m_Sigma[k]);
                    m_muG[k] = -(-m_rho * (m_mu[k] - Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                            + Utils.sumOfColumn(d.m_phi, k) - N * zeta_stat * moment);//-1 because LBFGS is minimization
                    fValue += -(-0.5 * m_rho * (m_mu[k] * m_mu[k]
                            - 2 * m_mu[k] * Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                            + m_mu[k] * Utils.sumOfColumn(d.m_phi, k) - N * zeta_stat * moment);
                }
                LBFGS.lbfgs(number_of_topics, 4, m_mu, fValue, m_muG, false, mu_diag, iprint, 1e-2, 1e-32, iflag);
//                System.out.println("-- update mu: fValue: " + fValue + "; mu: "
//                        + m_mu[0] + "; gradient: " + m_muG[0]);
            } while (iter++ < iterMax && iflag[0] != 0);
        }catch (ExceptionWithIflag e){
            e.printStackTrace();
        }
        for(int k = 0; k < number_of_topics; k++){
            d.m_mu[k] = m_mu[k];
        }
    }

    // alternative: line search / fixed-stepsize gradient descent
    public void update_mu(_Doc d, _User u, _Product i){
        int[] iflag = {1}, iprint = {-1,3};
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double last = 1.0;
        double[] cur = new double[number_of_topics];
        double stepsize = 1e-1, alpha = 1e-4, c2 = 0.5, beta = 0.8;
        double[] m_diagG = new double[number_of_topics];
        double[] m_muG = new double[number_of_topics]; // gradient for mu
        double[] mu_diag = new double[number_of_topics];
        double muG2Norm = 0.0, diagG2Norm = 0.0;
        int N = d.getTotalDocLength();

        double[] m_phiStat = new double[number_of_topics];
        Arrays.fill(m_phiStat, 0.0);
        _SparseFeature[] fv = d.getSparse();
        for(int k = 0;k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                int wid = fv[n].getIndex();
                double v = fv[n].getValue();
                m_phiStat[k] += v * d.m_phi[n][k];
            }
        }

        Arrays.fill(mu_diag, 0.0);

        double moment, zeta_stat = 1.0 / d.m_zeta;
        do {
            //update gradient of mu
            lastFValue = fValue;
            fValue = 0.0;
            for (int k = 0; k < number_of_topics; k++) {
                stepsize = 0.5;
                moment = Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
                m_muG[k] = -(-m_rho * (d.m_mu[k] - Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                        + m_phiStat[k] - N * zeta_stat * moment);//-1 because LBFGS is minimization
                last = - ( -0.5 * m_rho * (d.m_mu[k] * d.m_mu[k]
                        - 2 * d.m_mu[k] * Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                        + d.m_mu[k] * m_phiStat[k] - N * zeta_stat * moment);
                //line search
                do{
                    stepsize = beta * stepsize;
                    mu_diag[k] = d.m_mu[k] - stepsize * m_muG[k];
                    moment = Math.exp(mu_diag[k] + 0.5 * mu_diag[k]);
                    m_diagG[k] = -(-m_rho * (mu_diag[k] - Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                            + m_phiStat[k] - N * zeta_stat * moment);
                    cur[k] = - ( -0.5 * m_rho * (mu_diag[k] * mu_diag[k]
                            - 2 * mu_diag[k] * Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                            + mu_diag[k] * m_phiStat[k] - N * zeta_stat * moment);
                    diff = cur[k] - last;
                    muG2Norm = Math.pow(m_muG[k],2);
                    diagG2Norm = Math.pow(m_diagG[k], 2);
                    
                }while(diff > - alpha * stepsize * muG2Norm && muG2Norm > diagG2Norm);
                d.m_mu[k] = mu_diag[k];

                //fix stepsize
//                d.m_mu[k] = d.m_mu[k] - stepsize * m_muG[k];
                moment = Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
                fValue += - ( -0.5 * m_rho * (d.m_mu[k] * d.m_mu[k]
                        - 2 * d.m_mu[k] * Utils.dotProduct(i.m_eta, u.m_nuP[k]) / Utils.sumOfArray(i.m_eta))
                        + d.m_mu[k] * m_phiStat[k] - N * zeta_stat * moment);
            }
//                LBFGS.lbfgs(number_of_topics,4, d.m_mu, fValue, m_muG,false, mu_diag, iprint, 1e-6, 1e-16, iflag);
            diff = Math.abs((lastFValue - fValue) / lastFValue);
//            System.out.println("-- update mu: fValue: " + fValue + "; diff: " + diff
//                    + "; gradient: " + Utils.dotProduct(m_muG, m_muG) + "; mu: " + d.m_mu[0]);
        } while (iter++ < iterMax && diff > cvg);
    }

    //variational inference for p(\theta|\mu,\Sigma) for each document: Quasi-Newton, LBFGS
    public void update_SigmaTheta1(_Doc d){
        int[] iflag = {1}, iprint = {-1,3};
        double fValue = 1.0;
        int N = d.getTotalDocLength();
        double[] m_sigmaSqrt = new double[number_of_topics];
        double[] m_SigmaG = new double[number_of_topics]; // gradient for Sigma
        double[] sigma_diag = new double[number_of_topics];
        Arrays.fill(sigma_diag, 0.0);

//        double[] log_sigma = new double[number_of_topics];
//        for(int k=0; k < number_of_topics; k++){
//            log_sigma[k] = Math.log(d.m_Sigma[k]);
//        }

        for(int k = 0; k < number_of_topics; k++){
            m_sigmaSqrt[k] = Math.sqrt(d.m_Sigma[k]);
        }

        double moment, sigma;
        int iter = 0, iterMax = 10;
        try {
            do {
                fValue = 0.0;
                //update gradient of sigma
                for (int k = 0; k < number_of_topics; k++) {
                    sigma = Math.pow(m_sigmaSqrt[k], 2);
                    moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                    m_SigmaG[k] = - (1.0 / m_sigmaSqrt[k] - m_rho * m_sigmaSqrt[k] - m_sigmaSqrt[k] * N  * moment / d.m_zeta); //-1 because LBFGS is minimization
                    fValue += - (-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
                }
                System.out.println("-- update sigmaTheta: fValue: " + fValue+ "; sigmaTheta: "
                        + m_sigmaSqrt[0] + "; gradient: " + Utils.dotProduct(m_SigmaG, m_SigmaG));
                LBFGS.lbfgs(number_of_topics, 4, m_sigmaSqrt, fValue, m_SigmaG, false, sigma_diag, iprint, 1e-2, 1e-32, iflag);

            } while (iter++ < iterMax && iflag[0] != 0);
        }catch (ExceptionWithIflag e){
            e.printStackTrace();
        }
        for(int k = 0;k < number_of_topics; k++){
            d.m_Sigma[k] = Math.pow(m_sigmaSqrt[k], 2);
        }
    }

    public void update_SigmaTheta(_Doc d){
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double last = 1.0;
        double cur = 0.0;
        double stepsize , alpha = 0.5, beta = 0.8;
        int N = d.getTotalDocLength();
        double[] m_SigmaG = new double[number_of_topics]; // gradient for Sigma
        double[] m_SigmaGl = new double[number_of_topics];
        double[] m_sigmaNew = new double[number_of_topics];
        Arrays.fill(m_sigmaNew, 0.0);

        double[] m_sigmaSqrt = new double[number_of_topics];
        for(int k=0; k < number_of_topics; k++){
            m_sigmaSqrt[k] = Math.sqrt(d.m_Sigma[k]);
        }

        double moment, sigma;
        double sigma2Norm=0.0, diag2Norm = 0.0;
        do {
            //update gradient of sigma
            last = 0.0;
            for (int k = 0; k < number_of_topics; k++) {
                sigma = Math.pow(m_sigmaSqrt[k], 2);
                moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                m_SigmaG[k] = -(-m_rho * m_sigmaSqrt[k] - N * m_sigmaSqrt[k] * moment / d.m_zeta + 1.0 / m_sigmaSqrt[k]); //-1 because LBFGS is minimization
                last += -(-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
            }

            //line search
            stepsize = 1.0;
            int iterLS = 10, i=0;
            do{
                cur = 0.0;
                stepsize = beta * stepsize;
                for (int k = 0; k < number_of_topics; k++) {
                    m_sigmaNew[k] = m_sigmaSqrt[k] - stepsize * m_SigmaG[k];
                    sigma = Math.pow(m_sigmaNew[k], 2);
                    moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                    m_SigmaGl[k] = -(-m_rho * m_sigmaNew[k] - N * moment * m_sigmaNew[k] / d.m_zeta + 1.0 / m_sigmaNew[k]);
                    cur += -(-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
                }
                diff = cur - last;
                sigma2Norm = Utils.dotProduct(m_SigmaG, m_SigmaG);
                diag2Norm = Utils.dotProduct(m_SigmaGl, m_SigmaGl);
                i += 1;
//                System.out.println("---- line search for sigmaTheta: cur: "
//                        + cur + "; diff: " + diff + "; gradient: " + diag2Norm);
                if(i > iterLS && diff <= 0.0){
                    break;
                }
            }while(diff > - alpha * stepsize * sigma2Norm && diag2Norm > sigma2Norm);

            lastFValue = fValue;
            fValue = 0.0;
            for(int k = 0; k < number_of_topics;k ++) {
                m_sigmaSqrt[k] = m_sigmaSqrt[k] - stepsize * m_SigmaG[k];
                sigma = Math.pow(m_sigmaSqrt[k], 2);
                moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                fValue += -(-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
            }

//            LBFGS.lbfgs(number_of_topics,4, d.m_Sigma, fValue, m_SigmaG,false, sigma_diag, iprint, 1e-6, 1e-32, iflag);

            diff = Math.abs((lastFValue - fValue) / lastFValue);
//            System.out.println("-- update sigmaTheta: fValue: " + fValue + "; diff: " + diff
//                    + "; gradient: " + Utils.dotProduct(m_SigmaG, m_SigmaG) + "; sigma: " + Math.pow(m_sigmaSqrt[0], 2));
        } while(iter++ < iterMax && diff > cvg);

        for(int k=0; k < number_of_topics; k++){
            d.m_Sigma[k] = Math.pow(m_sigmaSqrt[k], 2);
        }
//        System.out.println("sigmasum: " + Utils.sumOfArray(d.m_Sigma));

    }

    //variational inference for p(P|\nu,\Sigma) for each user
    public void update_SigmaP(_User u){
        RealMatrix eta_stat_sigma = MatrixUtils.createRealIdentityMatrix(number_of_topics).scalarMultiply(m_sigma);
        for (int item_i = 0; item_i < number_of_items; item_i++) {
            RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(m_items[item_i].m_eta);
            double eta_0 = Utils.sumOfArray(m_items[item_i].m_eta);
            RealMatrix eta_stat_i = MatrixUtils.createRealDiagonalMatrix(m_items[item_i].m_eta).add(eta_vec.multiply(eta_vec.transpose()));
            eta_stat_sigma = eta_stat_sigma.add(eta_stat_i.scalarMultiply(m_rho / (eta_0 * (eta_0 + 1.0))));
        }
//        System.out.println("-- sigmaP before inverse: " + Arrays.toString(eta_stat_sigma.getColumn(1)));
        eta_stat_sigma = new LUDecomposition(eta_stat_sigma).getSolver().getInverse();
//        System.out.println("-- update sigmaP: " + Arrays.toString(eta_stat_sigma.getColumn(1)));
        for (int k = 0; k < number_of_topics; k++) {
            u.m_SigmaP[k] = eta_stat_sigma.getData();
        }
//        System.out.println("-- update sigmaP: now: " + Arrays.toString(u.m_SigmaP[0][0]));
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    public void update_nu(_User u, _Doc d){
        RealMatrix eta_stat_sigma = MatrixUtils.createRealMatrix(u.m_SigmaP[0]);
        RealMatrix eta_stat_nu = MatrixUtils.createColumnRealMatrix(new double[number_of_topics]);

//        System.out.println("-- update nuP: origin: " + Arrays.toString(u.m_nuP[0]));
        for (int k = 0; k < number_of_topics; k++) {
            for (int item_i = 0; item_i < number_of_items; item_i++) {
                RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(m_items[item_i].m_eta);
                double eta_0 = Utils.sumOfArray(m_items[item_i].m_eta);
                eta_stat_nu.add(eta_vec.scalarMultiply(d.m_mu[k] / eta_0));
            }
            u.m_nuP[k] = eta_stat_sigma.multiply(eta_stat_nu).scalarMultiply(m_rho).getColumn(0);
        }
//        System.out.println("-- update nuP: origin: " + Arrays.toString(u.m_nuP[0]));
    }

    public void update_eta1(_Product i, _Doc d){
        int[] iflag = {0}, iprint = {-1,3};
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double last = 1.0;
        double cur;
        double stepsize = 1e-1, alpha = 0.5, beta = 0.8;
        double[] m_etaG = new double[number_of_topics];
        double[] eta_log = new double[number_of_topics];
        double[] eta_temp = new double[number_of_topics];
        double[] eta_diag = new double[number_of_topics];
        double[] eta_diag_log = new double[number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            eta_log[k] = Math.log(i.m_eta[k]);
            eta_temp[k] = i.m_eta[k];
        }


        double eta_0, eta_0_square = 0.0;
        do{
            lastFValue = fValue;
            last = 0.0;
            stepsize = 0.0025;
            eta_0_square = 0.0;
            for(int k=0;k < number_of_topics; k++){
                eta_0_square += Math.pow(eta_temp[k], 2);
            }
            eta_0 = Utils.sumOfArray(eta_temp);

            for(int k = 0; k < number_of_topics; k++) {
                //might be optimized using global stats
                double gTerm1 = 0.0;
                double gTerm2 = 0.0;
                double gTerm3 = 0.0;
                double gTerm4 = 0.0;
                double term1 = 0.0;
                double term2 = 0.0;
                for (int uid = 0; uid < number_of_users; uid++) {
                    for (int j = 0; j < number_of_topics; j++) {
                        gTerm1 += m_users[uid].m_nuP[j][k] * d.m_mu[j];
                        term1 += eta_temp[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];

                        for (int l = 0; l < number_of_topics; l++) {
                            gTerm2 += eta_temp[l] * m_users[uid].m_nuP[j][l] * d.m_mu[j];

                            gTerm3 += eta_temp[k] * eta_temp[l] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                gTerm3 += (Math.pow(eta_temp[k], 2) + eta_temp[k]) * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            }

                            term2 += eta_temp[l] * eta_temp[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                term2 += eta_temp[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
                            }

                            for (int p = 0; p < number_of_topics; p++) {
                                gTerm4 += eta_temp[l] * eta_temp[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                if (p == l) {
                                    gTerm4 += eta_temp[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                }
                            }
                        }
                    }
                }
                m_etaG[k] = -(Utils.trigamma(eta_temp[k]) * (m_alpha[k] * eta_temp[k] - Math.pow(eta_temp[k], 2))
                        - Utils.trigamma(eta_0) * (Utils.sumOfArray(m_alpha) - eta_0_square)
                        + m_rho * eta_temp[k] * gTerm1 / eta_0 - m_rho * eta_temp[k] * gTerm2 / (Math.pow(eta_0, 2))
                        - m_rho * gTerm3 / (2 * eta_0 * (eta_0 + 1.0))
                        + m_rho * (2 * eta_0 + 1.0) * eta_temp[k] * gTerm4 / (2 * Math.pow(eta_0, 2) * Math.pow(eta_0 + 1.0, 2)));

                last += -((m_alpha[k] - eta_temp[k]) * (Utils.digamma(eta_temp[k]) - Utils.digamma(eta_0))
                        - Utils.lgamma(eta_0) + Utils.lgamma(eta_temp[k])
                        + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
            }
//            do{
//                stepsize = beta * stepsize;
//                for(int k = 0; k < number_of_topics; k++) {
//                    eta_diag_log[k] = eta_log[k] - stepsize * m_etaG[k];
//                    eta_diag[k] = Math.exp(eta_diag_log[k]);
//                }
//                double eta0_diag = Utils.sumOfArray(eta_diag);
//                double term1_diag = 0.0;
//                double term2_diag = 0.0;
//                cur = 0.0;
//                for(int k = 0; k < number_of_topics; k++) {
//                    for (int uid = 0; uid < number_of_users; uid++) {
//                        for (int j = 0; j < number_of_topics; j++) {
//                            term1_diag += eta_diag[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];
//                            for (int l = 0; l < number_of_topics; l++) {
//                                term2_diag += eta_diag[l] * eta_diag[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
//                                if (l == k) {
//                                    term2_diag += eta_diag[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
//                                }
//                            }
//                        }
//                    }
//                    cur += -((m_alpha[k] - eta_diag[k]) * (Utils.digamma(eta_diag[k]) - Utils.digamma(eta0_diag))
//                            - Utils.lgamma(eta0_diag) + Utils.lgamma(eta_diag[k])
//                            + m_rho * term1_diag / eta0_diag - m_rho * term2_diag / (2 * eta0_diag * (eta0_diag + 1.0)));
//                }
//                diff = cur - last;
////                System.out.println("----  line search: cur: " + cur + "; diff: " + diff
////                            + "; eta_diag: " + eta_diag[0]
////                            + "; etaG: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta0_diag: " + eta0_diag
////                            + "; stepsize: " + stepsize);
//
//            }while(diff > - alpha * stepsize * Utils.dotProduct(m_etaG, m_etaG));
//            fValue = cur;


            fValue = last;
            for(int k = 0; k < number_of_topics; k++) {
                eta_diag_log[k] = eta_log[k] - stepsize * m_etaG[k];
                eta_diag[k] = Math.exp(eta_diag_log[k]);
            }

            for(int k = 0; k < number_of_topics; k++) {
                eta_log[k] = eta_diag_log[k];
                eta_temp[k] = Math.exp(eta_log[k]);
            }

            diff = Math.abs((lastFValue - fValue) / lastFValue);
//            System.out.println("-- update eta: fValue: " + fValue + "; diff: " + diff
//                    + "; gradient: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta: " + Utils.sumOfArray(eta_temp));
//            LBFGS.lbfgs(number_of_topics,4, i.m_eta, fValue, m_etaG,false, eta_diag, iprint, 1e-6, 1e-32, iflag);
        }while(iter++ < iterMax && diff > cvg);

        for(int k=0;k<number_of_topics;k++){
            i.m_eta[k] = eta_temp[k];
        }
    }

    public void update_eta(_Product i, _Doc d){
        int[] iflag = {0}, iprint = {-1,3};
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double last = 1.0;
        double cur;
        double stepsize = 1e-1, alpha = 0.5, beta = 0.8;
        double[] m_etaG = new double[number_of_topics];
        double[] eta_diag = new double[number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            eta_diag[k] = i.m_eta[k];
        }

        double eta_0, monitorNeg = 0.0;
        do{
            lastFValue = fValue;
            last = 0.0;
            monitorNeg = 0.0;
            stepsize = 0.25;
            for(int k = 0; k < number_of_topics; k++) {

                //might be optimized using global stats
                eta_0 = Utils.sumOfArray(i.m_eta);
                double gTerm1 = 0.0;
                double gTerm2 = 0.0;
                double gTerm3 = 0.0;
                double gTerm4 = 0.0;
                double term1 = 0.0;
                double term2 = 0.0;
                for (int uid = 0; uid < number_of_users; uid++) {
                    for (int j = 0; j < number_of_topics; j++) {
                        gTerm1 += m_users[uid].m_nuP[j][k] * d.m_mu[j];
                        term1 += i.m_eta[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];

                        for (int l = 0; l < number_of_topics; l++) {
                            gTerm2 += i.m_eta[l] * m_users[uid].m_nuP[j][l] * d.m_mu[j];

                            gTerm3 += i.m_eta[l] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                gTerm3 += (i.m_eta[l] + 1.0) * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            }

                            term2 += i.m_eta[l] * i.m_eta[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                term2 += i.m_eta[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
                            }

                            for (int p = 0; p < number_of_topics; p++) {
                                gTerm4 += i.m_eta[l] * i.m_eta[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                if (p == l) {
                                    gTerm4 += i.m_eta[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
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
                if(k == 0) {
                    double eps = 0.01;
                    i.m_eta[k] = i.m_eta[k] + eps;
                    double post = -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                            - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
                    i.m_eta[k] = i.m_eta[k] - eps;
                    double pre = -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                            - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
                    double check = (post - pre) / eps;
                    System.out.println("-- check gradient: " + check);
                }


                last += -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                        - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                        + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
            }
            do{
                stepsize = beta * stepsize;
                for(int k = 0; k < number_of_topics; k++) {
                    eta_diag[k] = i.m_eta[k] - stepsize * m_etaG[k];
                }
                double eta0_diag = Utils.sumOfArray(eta_diag);
                double term1_diag = 0.0;
                double term2_diag = 0.0;
                cur = 0.0;
                for(int k = 0; k < number_of_topics; k++) {
                    for (int uid = 0; uid < number_of_users; uid++) {
                        for (int j = 0; j < number_of_topics; j++) {
                            term1_diag += eta_diag[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];
                            for (int l = 0; l < number_of_topics; l++) {
                                term2_diag += eta_diag[l] * eta_diag[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                                if (l == k) {
                                    term2_diag += eta_diag[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
                                }
                            }
                        }
                    }
                    cur += -((m_alpha[k] - eta_diag[k]) * (Utils.digamma(eta_diag[k]) - Utils.digamma(eta0_diag))
                            - Utils.lgamma(eta0_diag) + Utils.lgamma(eta_diag[k])
                            + m_rho * term1_diag / eta0_diag - m_rho * term2_diag / (2 * eta0_diag * (eta0_diag + 1.0)));
                }
                diff = cur - last;
//                System.out.println("----  line search: cur: " + cur + "; diff: " + diff
//                            + "; eta_diag: " + eta_diag[0]
//                            + "; etaG: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta0_diag: " + eta0_diag
//                            + "; stepsize: " + stepsize);

            }while(diff > - alpha * stepsize * Utils.dotProduct(m_etaG, m_etaG));
            for(int k = 0; k < number_of_topics; k++) {
                i.m_eta[k] = eta_diag[k];
                if (i.m_eta[k] <= 0){
                    monitorNeg += i.m_eta[k];
                }
            }

            fValue = cur;
            diff = Math.abs((lastFValue - fValue) / lastFValue);
//            System.out.println("-- update eta: fValue: " + fValue + "; diff: " + diff
//                    + "; gradient: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta: " + Utils.sumOfArray(i.m_eta)
//                    + "; monitor: " + monitorNeg);
//            LBFGS.lbfgs(number_of_topics,4, i.m_eta, fValue, m_etaG,false, eta_diag, iprint, 1e-6, 1e-32, iflag);
        }while(iter++ < iterMax && diff > cvg);
    }

    public void update_eta2(_Product i, _Doc d){
        int[] iflag = {0}, iprint = {-1,3};
        int iterMax = 10, iter = 0;
        double fValue;
        double[] m_eta = new double[number_of_topics];
        double[] m_etaG = new double[number_of_topics];
        double[] eta_diag = new double[number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            eta_diag[k] = i.m_eta[k];
            m_eta[k] = i.m_eta[k];
        }

        double eta_0=0;
        try {
            do {
                fValue = 0;
                for (int k = 0; k < number_of_topics; k++) {

                    //might be optimized using global stats
                    eta_0 = Utils.sumOfArray(m_eta);
                    double gTerm1 = 0.0;
                    double gTerm2 = 0.0;
                    double gTerm3 = 0.0;
                    double gTerm4 = 0.0;
                    double term1 = 0.0;
                    double term2 = 0.0;
                    for (int uid = 0; uid < number_of_users; uid++) {
                        for (int j = 0; j < number_of_topics; j++) {
                            gTerm1 += m_users[uid].m_nuP[j][k] * d.m_mu[j];
                            term1 += m_eta[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];

                            for (int l = 0; l < number_of_topics; l++) {
                                gTerm2 += m_eta[l] * m_users[uid].m_nuP[j][l] * d.m_mu[j];

                                gTerm3 += m_eta[l] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                                if (l == k) {
                                    gTerm3 += (m_eta[l] + 1.0) * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                                }

                                term2 += m_eta[l] * m_eta[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                                if (l == k) {
                                    term2 += m_eta[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
                                }

                                for (int p = 0; p < number_of_topics; p++) {
                                    gTerm4 += m_eta[l] * m_eta[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                    if (p == l) {
                                        gTerm4 += m_eta[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                    }
                                }
                            }
                        }
                    }
                    m_etaG[k] = -(Utils.trigamma(m_eta[k]) * (m_alpha[k] - m_eta[k])
                            - Utils.trigamma(eta_0) * (Utils.sumOfArray(m_alpha) - eta_0)
                            + m_rho * gTerm1 / eta_0 - m_rho * gTerm2 / (Math.pow(eta_0, 2))
                            - m_rho * gTerm3 / (2 * eta_0 * (eta_0 + 1.0))
                            + m_rho * (2 * eta_0 + 1.0) * gTerm4 / (2 * Math.pow(eta_0, 2) * Math.pow(eta_0 + 1.0, 2)));

                    fValue += -((m_alpha[k] - m_eta[k]) * (Utils.digamma(m_eta[k]) - Utils.digamma(eta_0))
                            - Utils.lgamma(eta_0) + Utils.lgamma(m_eta[k])
                            + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
                }
                LBFGS.lbfgs(number_of_topics,4, m_eta, fValue, m_etaG,false, eta_diag, iprint, 1e-6, 1e-32, iflag);
//                System.out.println("-- update eta: fValue: " + fValue + "; eta: " + m_eta[0]
//                        + "; gradient: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta0: " + eta_0 );
            } while (iter++ < iterMax && iflag[0] != 0);
        }catch(ExceptionWithIflag e){
            e.printStackTrace();
        }

        for(int k = 0; k < number_of_topics; k++){
            i.m_eta[k] = m_eta[k];
        }
    }



    public void M_step() {
        //maximize likelihood for \rho of p(\theta|P\gamma, \rho)
        double term1 = m_thetaStats;
        double term2 = Utils.sumOfArray(m_eta_p_Stats);
        double term3 = Utils.sumOfArray(m_eta_mean_Stats) * 2;
        m_rho = number_of_topics / (term1 + term2 - term3);

        //maximize likelihood for \sigma
        m_sigma = number_of_topics / m_pStats;

        //maximize likelihood for \beta
        for(int k = 0 ;k < number_of_topics; k++){
            double sum = Utils.sumOfArray(m_word_topic_stats[k]);
            for(int v = 0; v < vocabulary_size; v++){
                m_beta[k][v] = m_word_topic_stats[k][v] / sum;
            }
        }

        //maximize likelihood for \alpha using Newton
        int i = 0;
        double diff = 0.0, alphaSum, diAlphaSum, z, c1, c2, c, deltaAlpha;
        do{
            alphaSum = Utils.sumOfArray(m_alpha);
            diAlphaSum = Utils.digamma(alphaSum);
            z = number_of_items * Utils.trigamma(alphaSum);

            c1 = 0; c2 = 0;
            for(int k = 0; k < number_of_topics; k++){
                m_alphaG[k] = number_of_items * (diAlphaSum - Utils.digamma(m_alpha[k])) + m_etaStats[k];
                m_alphaG[k] = - number_of_items * Utils.trigamma(m_alpha[k]);

                c1 += m_alphaG[k] / m_alphaH[k];
                c2 += 1.0 / m_alphaH[k];
            }
            c = c1 / (1.0/z + c2);

            diff = 0.0;
            for(int k = 0; k < number_of_topics; k++){
                deltaAlpha = (m_alphaG[k] -c) / m_alphaH[k];
                m_alpha[k] -= deltaAlpha;
                diff += deltaAlpha * deltaAlpha;
            }
            diff /= number_of_topics;
        }while(++i < m_varMaxIter && diff > m_varConverge);

    }

    protected double calc_log_likelihood(_Doc doc, _User currentU, _Product currentI) {

        double log_likelihood = 0.0;
        double eta0 = Utils.sumOfArray(currentI.m_eta);
        double diGammaEtaSum = Utils.digamma(eta0);
        double lgammaEtaSum = Utils.lgamma(eta0);
        double lgammaAlphaSum = Utils.lgamma(Utils.sumOfArray(m_alpha));


        //part1 (term1-term6)
        double part1 = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            part1 += (m_alpha[k] - currentI.m_eta[k]) * (Utils.digamma(currentI.m_eta[k]) - diGammaEtaSum);
            part1 -= Utils.lgamma(m_alpha[k]) - Utils.lgamma(currentI.m_eta[k]);
        }
        part1 += lgammaAlphaSum - lgammaEtaSum;
        log_likelihood += part1;
//        System.out.println("part1: " + part1);

        // part2 (term2-term7)
        double part2 = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            double temp1 = 0.0;
            for(int l = 0; l < number_of_topics; l++) {
                temp1 += currentU.m_SigmaP[k][l][l] + currentU.m_nuP[k][l] * currentU.m_nuP[k][l];
            }
            double det = new LUDecomposition(MatrixUtils.createRealMatrix(currentU.m_SigmaP[k])).getDeterminant();
            part2 += -0.5 * (temp1 * m_sigma - number_of_topics)
                    + 0.5 * (number_of_topics * Math.log(m_sigma) + Math.log(det));
        }
        log_likelihood += part2;
//        System.out.println("part2: " + part2);


        //part3 (term3-term8)
        double term1 = 0.0;
        double term2 = 0.0;
        double term3 = 0.0;
        double term4 = 0.0;
        double part3 = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            term1 += doc.m_Sigma[k] + doc.m_mu[k] * doc.m_mu[k];
            for(int j = 0; j < number_of_topics; j++){
                term2 += currentI.m_eta[k] * currentU.m_nuP[j][k] * doc.m_mu[j];

                for(int l = 0; l < number_of_topics; l++){
                    term3 += currentI.m_eta[j] * currentI.m_eta[l] *
                            (currentU.m_SigmaP[k][j][l] + currentU.m_nuP[k][j] * currentU.m_nuP[k][l]);
                    if(l == j){
                        term3 += currentI.m_eta[l] *
                                (currentU.m_SigmaP[k][j][l] + currentU.m_nuP[k][j] * currentU.m_nuP[k][l]);
                    }
                }
            }
            term4 += Math.log(m_rho * doc.m_Sigma[k]);
        }
        part3 += -m_rho * (0.5 * term1 - 2 * term2 / eta0 + term3 / (eta0 * (eta0 + 1.0))) + number_of_topics/2.0
                + 0.5 * term4;
        log_likelihood += part3;
//        System.out.println("part3: " + part3 + "; eta0: " + eta0 + "; term1: " + term1
//                + "; term2: " + term2 + "; term3: " + term3 + "; term4: " + term4 + "; sigmaSum: " + Utils.sumOfArray(doc.m_Sigma));

        //part4
        int wid;
        double v;
        double part4 = 0.0, part5 = 0.0;
        term1 = 0.0;
        term2 = 0.0;
        term3 = 0.0;
        _SparseFeature[] fv = doc.getSparse();
//        System.out.println("file length: " + fv.length);
        for(int k = 0; k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                term1 += v * doc.m_phi[n][k] * doc.m_mu[k];
                term3 += v * doc.m_phi[n][k] * Math.log(doc.m_phi[n][k]);
                part5 += v * doc.m_phi[n][k] * Math.log(m_beta[k][wid]);
            }
            term2 += Math.exp(doc.m_mu[k] + doc.m_Sigma[k]/2.0);
        }
        part4 += term1 - term2 / doc.m_zeta + 1.0 - Math.log(doc.m_zeta) - term3;
//        System.out.println("part4: " + part4 + "; term1: " + term1
//                + "; term2: " + term2 + "; term3: " + term3 + "; zeta" + doc.m_zeta);
        log_likelihood += part4;
//        System.out.println("part5: " + part5);
        log_likelihood += part5;

        return log_likelihood;
    }

    public void EM(){
        System.out.println("Initializing model...");
        initModel();

        System.out.println("Initializing documents...");
        for(_Doc doc : m_corpus.getCollection()){
            initDoc(doc);
        }
        System.out.println("Initializing users...");
        for(_User user : m_users){
            initUser(user);
        }
        System.out.println("Initializing items...");
        for(_Product item : m_items){
            initItem(item);
        }

        int iter = 0;
        double lastAllLikelihood = 0.0;
        double currentAllLikelihood = 0.0;
        double converge = 0.0;
        do{
            currentAllLikelihood = E_step();
            if(iter > 0){
                converge = (lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood;
            }else{
                converge = 1.0;
            }

            if(converge < 0){
                m_varMaxIter += 10;
                System.out.println("! E_step not converge...");
            }else{
                M_step();
                lastAllLikelihood = currentAllLikelihood;
                System.out.format("%s step: likelihood is %.3f, converge to %f...\n",
                        iter, currentAllLikelihood, converge);
                iter++;
                if(converge < m_emConverge)
                    break;
            }
            initStats();
        }while(iter < m_emMaxIter && converge > m_emConverge);
    }

}
