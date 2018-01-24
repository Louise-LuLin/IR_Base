package topicmodels;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import structures._Corpus;
import structures._Review;
import structures._User;
import structures._Item;
import structures._SparseFeature;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class Topic_ItemRecmd extends LDA_Variational {

    protected double m_rho;
    protected double m_sigma;

    protected double[] m_muG; // gradient for mu
    protected double[] m_muH; // Hessian for mu
    protected double[] m_SigmaG; // gradient for Sigma
    protected double[] m_SigmaH; // Hessian for Sigma

    protected int number_of_users;
    protected int number_of_items;


    public Topic_ItemRecmd(int number_of_iteration, double converge,  int varMaxIter, double varConverge, //stop criterion
                           _Corpus c, int number_of_topics, int number_of_items, int number_of_users, //user pre-defined arguments
                           double lambda, double beta, double alpha) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge);
        this.number_of_items = number_of_items;
        this.number_of_users = number_of_users;
    }

    @Override
    protected void createSpace() {
        super.createSpace();

        m_muG = new double[number_of_topics];
        m_muH = new double[number_of_topics];
        m_SigmaG = new double[number_of_topics];
        m_SigmaH = new double[number_of_topics];
    }

    @Override
    public double calculate_E_step(_Review d, _User u, _Item i) {
        double last = 1;
        if (m_converge > 0){
            last = calculate_log_likelihood(d, u ,i);
        }

        double current = last, converge, logSum, v;
        int iter = 0, wid;
        _SparseFeature[] fv = d.getSparse();

        do{
            //variational inference for p(z|w,\phi) for each document
            for(int n = 0; n < fv.length; n++){
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                for(int k = 0; k < number_of_topics; k++){
                    d.m_phi[n][k] = Math.log(topic_term_probabilty[k][wid]) + d.m_mu[k];
                }
                // normalize
                logSum = Utils.logSum(d.m_phi[n]);
                for(int k = 0; k < number_of_topics; k++){
                    d.m_phi[n][k] = Math.exp(d.m_phi[n][k]-logSum);
                }
            }

            //variational inference for p(\theta|\mu,\Sigma) for each document
            //estimate zeta
            d.m_zeta = 0;
            for(int k = 0; k < number_of_topics; k++){
                d.m_zeta += Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
            }

            //estimate \sigma using Newton method
            int iter_i = 0;
            double diff, delta, moment, zeta_stat = 1.0 / d.m_zeta;
            do{
                for(int k = 0; k < number_of_topics; k++){
                    moment = Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
                    m_SigmaG[k] = 0.5 * (1.0 / d.m_Sigma[k] - m_rho - zeta_stat * moment);
                    m_SigmaH[k] = -0.5 * (1.0 / d.m_Sigma[k] + 0.5 * zeta_stat * moment);
                }
                diff = 0;
                for(int k = 0; k < number_of_topics; k++){
                    delta = m_SigmaG[k] / m_SigmaH[k];
                    d.m_Sigma[k] -= delta;
                    diff += delta * delta;
                }
                diff /= number_of_topics;
            } while (++iter_i < m_varMaxIter && diff > m_varConverge);

            //estimate /mu using Newton method
            iter_i = 0;
            do{
                for(int k = 0; k < number_of_topics; k++){
                    moment = Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
                    m_muG[k] = -m_rho * (d.m_mu[k] - Utils.dotProduct(i.m_eta, u.m_nuP[k])/Utils.sumOfArray(i.m_eta))
                            + Utils.sumOfColumn(d.m_phi,k) - zeta_stat * moment;
                    m_muH[k] = -m_rho - zeta_stat * moment;
                }
                diff = 0;
                for(int k = 0; k < number_of_topics; k++){
                    delta = m_muG[k] / m_muH[k];
                    d.m_mu[k] -= delta;
                    diff += delta * delta;
                }
                diff /= number_of_topics;
            } while(++iter_i < m_varMaxIter && diff > m_varConverge);

            //variational inference for p(P|\nu,\Sigma) for each user
            RealMatrix eta_stat_sigma = MatrixUtils.createRealIdentityMatrix(number_of_topics).scalarMultiply(m_sigma);
            RealMatrix eta_stat_nu = MatrixUtils.createColumnRealMatrix(new double[number_of_topics]);
            for(int item_i=0; item_i < number_of_items; item_i++){
                RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(i.m_eta);
                double eta_0 = Utils.sumOfArray(i.m_eta);
                RealMatrix eta_stat_i = MatrixUtils.createRealDiagonalMatrix(i.m_eta).add(eta_vec.multiply(eta_vec.transpose()));
                eta_stat_sigma.add(eta_stat_i.scalarMultiply(m_rho/(eta_0 * (eta_0 + 1.0))));
            }
            eta_stat_sigma = new LUDecomposition(eta_stat_sigma).getSolver().getInverse();
            for(int k = 0; k < number_of_topics; k++){
                u.m_SigmaP[k] = eta_stat_sigma.getData();
            }

            for(int k = 0; k < number_of_topics; k++){
                for (int item_i = 0; item_i < number_of_items; item_i++){
                    RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(i.m_eta);
                    double eta_0 = Utils.sumOfArray(i.m_eta);
                    eta_stat_nu.add(eta_vec.scalarMultiply(d.m_mu[k] / eta_0));
                }
                u.m_nuP[k] = eta_stat_sigma.multiply(eta_stat_nu).scalarMultiply(m_rho).getColumn(0);
            }

            //variational inference for p(\gamma|\eta) for each item
            for(int k = 0; k < number_of_topics; k++){

            }

            if(m_converge > 0){
                current = calculate_log_likelihood(d,u,i);
                converge = Math.abs((current - last) / last);
                last = current;
                if(converge < m_converge){
                    break;
                }
            }

        }while(++iter < number_of_iteration);
    }

    @Override
    public void calculate_M_step(int iter) {
        super.calculate_M_step(iter);
    }

    @Override
    protected double calculate_log_likelihood(_Review d, _User u, _Item i) {
        return super.calculate_log_likelihood();
    }


}
