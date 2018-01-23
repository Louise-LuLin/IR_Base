package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

public class TopicBasedItemRecmd extends LDA_Variational{

    protected double m_sigma;
    protected double m_rho;

    public PrintWriter summaryWriter;
    public PrintWriter logWirter;
    public PrintWriter debugWriter;

    public TopicBasedItemRecmd(int number_of_iteration, double converge,
                               double beta, _Corpus c, double lambda,
                               int number_of_topics, double alpha, int varMaxIter, double varConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge);

    }

    public void setSummaryWriter(String path){
        System.out.println("Summary File Path: " + path);
        try{
            summaryWriter = new PrintWriter(new File(path));
        }catch(Exception e){
            System.err.println(path + "not found!");
        }
    }

    public void setDebugWriter(String path){
        System.out.println("Debug File Path: " + path);
        try{
            debugWriter = new PrintWriter(new File(path));
        }catch(Exception e){
            System.err.println(path + "not found!");
        }
    }

    public void setLogWriter(String path){
        System.out.println("Log File Path: " + path);
        try{
            logWirter = new PrintWriter(new File(path));
        }catch(Exception e){
            System.err.println(path + "not found!");
        }
    }

    public void closeWriters(){
        if(summaryWriter != null){
            summaryWriter.flush();
            summaryWriter.close();
        }

        if(debugWriter != null){
            debugWriter.flush();
            debugWriter.close();
        }

        if(logWirter != null){
            logWirter.flush();
            logWirter.close();
        }
    }

}
