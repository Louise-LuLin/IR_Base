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

//        /**
//         * generate vocabulary
//         */
//        double startProb = 0.5; // Used in feature selection, the starting point of the features.
//        double endProb = 0.999; // Used in feature selection, the ending point of the features.
//        int maxDF = -1, minDF = 1; // Filter the features with DFs smaller than this threshold.
//        String featureSelection = "DF";
//
//        String folder = "./myData/byUser/";
//        String suffix = ".json";
//        String stopwords = "./data/Model/stopwords.dat";
//        String pattern = String.format("%dgram_%s", Ngram, featureSelection);
//        String fvFile = String.format("data/Features/fv_%s_byUser_20.txt", pattern);
//        String fvStatFile = String.format("data/Features/fv_stat_%s_byUser_20.txt", pattern);
//        String vctFile = String.format("data/Fvs/vct_%s_byUser_20.dat", pattern);
//
////        /****Loading json files*****/
//        DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//        analyzer.LoadStopwords(stopwords);
//        analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//
////		/****Feature selection*****/
//        System.out.println("Performing feature selection, wait...");
//        analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, maxDF, minDF); //Select the features.
//        analyzer.SaveCVStat(fvStatFile);

        /**
         * model training
         */
        String fvFile = "./data/Features/fv_1gram_DF_byItem_1.txt";
        String reviewFolder = "./myData/byItem_1/";
        String suffix = ".json";

        DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
        analyzer.LoadDirectory(reviewFolder, suffix);

        _Corpus corpus = analyzer.getCorpus();
//        corpus.save2File("./myData/byUser/top20.dat");

        int vocabulary_size = corpus.getFeatureSize();
        int topic_number = 30;

        int varMaxIter = 5;
        double varConverge = 1e-6;

        int emMaxIter = 20;
        double emConverge = 1e-3;

        double alpha = 1 + 1e-2, beta = 1.0 + 1e-3, sigma = 1.0 + 1e-2, rho = 1.0 + 1e-2;

        ETBIR etbirModel = new ETBIR(emMaxIter, emConverge, varMaxIter, varConverge,
                topic_number, alpha, sigma, rho, beta);
        etbirModel.loadCorpus(corpus);
        etbirModel.EM();
    }
}
