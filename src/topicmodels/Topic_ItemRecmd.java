package topicmodels;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class Topic_ItemRecmd extends LDA_Variational {

    protected double m_rho;
    protected double m_sigma;

    protected double[] m_

    public Topic_ItemRecmd(int number_of_iteration, double converge,  int varMaxIter, double varConverge, //stop criterion
                           _Corpus c, int number_of_topics, //user pre-defined arguments
                           double lambda, double beta, double alpha) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge);
    }

    @Override
    protected double calculate_log_likelihood(_Doc d) {
        return super.calculate_log_likelihood();
    }
}
