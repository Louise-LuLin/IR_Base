package mains;

import Analyzer.DocAnalyzer;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;

import structures._Corpus;
import structures._Review;
import topicmodels.ETBIR;


/**
 * @author Lu Lin
 */
public class ETBIRMain {

    public static void main(String[] args) throws IOException, ParseException {
        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 1; //The default value is unigram.
        String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
        int norm = 0;//The way of normalization.(only 1 and 2)
        int lengthThreshold = 5; //Document length threshold
        String tokenModel = "./data/Model/en-token.bin";
        String fvFile = "./data/Features/yelp_features.txt";

        String reviewFile = "./mydata/byItem/857_4JNXUYY8wbaaDmk3BPzlWw.json";

        DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
        analyzer.LoadReviewByItem(reviewFile);

        _Corpus corpus = analyzer.getCorpus();
        int vocabulary_size = corpus.getFeatureSize();
        int topic_number = 30;

        int varMaxIter = 20;
        double varConverge = 1e-6;
        double varStepSize = 1e-4;

        int emMaxIter = 20;
        double emConverge = 1e-3;

        double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, sigma = 1.0, rho = 1.0;

        ETBIR etbirModel = new ETBIR(emMaxIter, emConverge, varMaxIter, varConverge, varStepSize,
                topic_number, alpha, sigma, rho, beta);
        etbirModel.loadCorpus(corpus);
        etbirModel.EM();
    }
}
