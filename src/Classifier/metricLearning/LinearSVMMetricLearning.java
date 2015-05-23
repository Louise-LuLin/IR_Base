package Classifier.metricLearning;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

public class LinearSVMMetricLearning extends GaussianFieldsByRandomWalk {
	
	public enum FeatureType {
		FT_diff,
		FT_cross
	}
	
	protected Model m_libModel;
	int m_bound;
	double m_contSamplingRate;
	
	HashMap<Integer, Integer> m_selectedFVs;
	boolean m_learningBased = true;
	FeatureType m_fvType = FeatureType.FT_diff;
	
	//Default constructor without any default parameters.
	public LinearSVMMetricLearning(_Corpus c, String classifier, double C, int bound){
		super(c, classifier, C);
		m_bound = bound;
		m_contSamplingRate = 0.001; // a conservative setting
	}
	
	public LinearSVMMetricLearning(_Corpus c, String classifier, double C, 
			double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean storeGraph, 
			int bound, double cSampleRate) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta, storeGraph);
		m_bound = bound;
		m_contSamplingRate = cSampleRate;
	}
	
	public void setMetricLearningMethod(boolean opt) {
		m_learningBased = opt;
	}

	@Override
	public String toString() {
		return "LinearSVM-based Metric Learning for " + super.toString();
	}
	
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		if (!m_learningBased) {
			_SparseFeature[] xi = di.getProjectedFv(), xj = dj.getProjectedFv(); 
			if (xi==null || xj==null)
				return 0;
			else
				return Math.exp(Utils.calculateSimilarity(xi, xj));
		} else {
			Feature[] fv = createLinearFeature(di, dj);
			if (fv == null)
				return 0;
			else
				return Math.exp(Linear.predictValue(m_libModel, fv));//to make sure this is positive
		}
	}
	
	@Override
	protected void init() {
		super.init();
		m_libModel = trainLibLinear(m_bound);
	}
	
	@Override
	protected void constructGraph(boolean createSparseGraph) {
		for(_Doc d:m_testSet)
			d.setProjectedFv(m_selectedFVs);
		
		super.constructGraph(createSparseGraph);
	}
	
	double argmaxW(double[] w, int start, int size) {
		double max = Math.abs(w[start]);
		int index = 0;
		for(int i=1; i<size; i++) {
			if (Math.abs(w[start+i]) > max) {
				max = Math.abs(w[start+i]);
				index = i;
			}
		}
		return w[start+index];
	}
	
	//using L1 SVM to select a subset of features
	void selFeatures(Collection<_Doc> trainSet, double C) {
		Feature[][] fvs = new Feature[trainSet.size()][];
		double[] y = new double[trainSet.size()];
		
		int fid = 0;
		for(_Doc d:trainSet) {
			fvs[fid] = Utils.createLibLinearFV(d);
			y[fid] = d.getYLabel();
			fid ++;
		}
		
		Problem libProblem = new Problem();
		libProblem.l = fid;
		libProblem.n = m_featureSize;
		libProblem.x = fvs;
		libProblem.y = y;
		m_libModel = Linear.train(libProblem, new Parameter(SolverType.L1R_L2LOSS_SVC, C, 0.001));//use L1 regularization to reduce the feature size
		
		m_selectedFVs = new HashMap<Integer, Integer>();
		double[] w = m_libModel.getWeights();
		for(int i=0; i<m_featureSize; i++) {
			for(int c=0; c<m_classNo; c++) {
				if (w[i*m_classNo+c]!=0) {//a non-zero feature
					m_selectedFVs.put(i, m_selectedFVs.size());
					break;
				}	
			}
		}
		System.out.format("Selecting %d non-zero features by L1 regularization...\n", m_selectedFVs.size());
		
		for(_Doc d:trainSet) 
			d.setProjectedFv(m_selectedFVs);
		
		if (m_debugOutput!=null) {
			try {
				for(int i=0; i<m_featureSize; i++) {
					if (m_selectedFVs.containsKey(i)) {
						m_debugWriter.write(String.format("%s(%.2f), ", m_corpus.getFeature(i), argmaxW(w, i*m_classNo, m_classNo)));
					}
				}
				m_debugWriter.write("\n");
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}
	
	//In this training process, we want to get the weight of all pairs of samples.
	public Model trainLibLinear(int bound){
		//creating feature projection first (this is done by choosing important SVM features)
		selFeatures(m_trainSet, 0.1);
		
		if (!m_learningBased)
			return null;
		else {
			int mustLink = 0, cannotLink = 0, label;
//			Random rand = new Random();
			
			MyPriorityQueue<Double> maxSims = new MyPriorityQueue<Double>(1000, true), minSims = new MyPriorityQueue<Double>(1000, false);
			//In the problem, the size of feature size is m'*m'. (m' is the reduced feature space by L1-SVM)
			Feature[] fv;
			ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
			ArrayList<Integer> targetArray = new ArrayList<Integer>();
			for(int i = 0; i < m_trainSet.size(); i++){//directly using m_trainSet should not be a good idea!
				_Doc di = m_trainSet.get(i);
				
				for(int j = i+1; j < m_trainSet.size(); j++){
					_Doc dj = m_trainSet.get(j);
					
					if(di.getYLabel() == dj.getYLabel())//start from the extreme case?  && (d1.getYLabel()==0 || d1.getYLabel()==4)
						label = 1;
					else if(Math.abs(di.getYLabel() - dj.getYLabel())>bound)
						label = 0;
					else
						continue;
					
					double sim = super.getSimilarity(di, dj);
					if ( (label==1 && !minSims.add(sim)) || (label==0 && !maxSims.add(sim)) )
							continue;
					else if ((fv=createLinearFeature(di, dj))==null)
							continue;
					else {
						featureArray.add(fv);
						targetArray.add(label);
						
						if (label==1)
							mustLink ++;
						else
							cannotLink ++;
					}
				}
			}
			System.out.format("Generating %d must-links and %d cannot-links.\n", mustLink, cannotLink);
			
			Feature[][] featureMatrix = new Feature[featureArray.size()][];
			double[] targetMatrix = new double[targetArray.size()];
			for(int i = 0; i < featureArray.size(); i++){
				featureMatrix[i] = featureArray.get(i);
				targetMatrix[i] = targetArray.get(i);
			}
			
			double C = 1.0, eps = 0.01;
			Parameter libParameter = new Parameter(SolverType.L2R_L1LOSS_SVC_DUAL, C, eps);
			
			Problem libProblem = new Problem();
			libProblem.l = targetMatrix.length;
			if (m_fvType == FeatureType.FT_diff)
				libProblem.n = m_selectedFVs.size() * (1+m_selectedFVs.size())/2;
			else if (m_fvType == FeatureType.FT_cross)
				libProblem.n = m_selectedFVs.size() * m_selectedFVs.size();
			else {
				System.err.println("Unknown feature type for svm-based metric learning!");
				System.exit(-1);
			}
			libProblem.x = featureMatrix;
			libProblem.y = targetMatrix;
			Model model = Linear.train(libProblem, libParameter);
			return model;
		}
	}
	
	Feature[] createLinearFeature(_Doc d1, _Doc d2){ 
		if (m_fvType==FeatureType.FT_diff)
			return createLinearFeature_diff(d1, d2);
		else if (m_fvType==FeatureType.FT_cross)
			return createLinearFeature_cross(d1, d2);
		else
			return null;
	}
	
	//Calculate the new sample according to two documents.
	//Since cross-product will be symmetric, we don't need to store the whole matrix 
	Feature[] createLinearFeature_diff(_Doc d1, _Doc d2){
		_SparseFeature[] fv1=d1.getProjectedFv(), fv2=d2.getProjectedFv();
		if (fv1==null || fv2==null)
			return null;
		
		_SparseFeature[] diffVct = Utils.diffVector(fv1, fv2);
		
		Feature[] features = new Feature[diffVct.length*(diffVct.length+1)/2];
		int pi, pj, spIndex=0;
		double value = 0;
		for(int i = 0; i < diffVct.length; i++){
			pi = diffVct[i].getIndex();
			
			for(int j = 0; j < i; j++){
				pj = diffVct[j].getIndex();
				
				//Currently, we use one dimension array to represent V*V features 
				value = 2 * diffVct[i].getValue() * diffVct[j].getValue(); // this might be too small to count
				features[spIndex++] = new FeatureNode(getIndex(pi, pj), value);
			}
			value = diffVct[i].getValue() * diffVct[i].getValue(); // this might be too small to count
			features[spIndex++] = new FeatureNode(getIndex(pi, pi), value);
		}
		
		return features;
	}
	
	Feature[] createLinearFeature_cross(_Doc d1, _Doc d2){
		_SparseFeature[] fv1=d1.getProjectedFv(), fv2=d2.getProjectedFv();
		if (fv1==null || fv2==null)
			return null;
		
		Feature[] features = new Feature[fv1.length*fv2.length];
		int pi, pj, spIndex=0;
		double value = 0;
		for(int i = 0; i < fv1.length; i++){
			pi = fv1[i].getIndex();
			
			for(int j = 0; j < fv2.length; j++){
				pj = fv2[j].getIndex();
				
				//Currently, we use one dimension array to represent V*V features 
				value = fv1[i].getValue() * fv2[j].getValue(); // this might be too small to count
				features[spIndex++] = new FeatureNode(1+pi*m_selectedFVs.size()+pj, value);
			}
		}
		
		return features;
	}
	
	int getIndex(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return 1+i*(i+1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
	
	public void verifySimilarity(String filename, boolean append) {
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename, append), "UTF-8"));
			
			for(_Doc d:m_testSet)
				d.setProjectedFv(m_selectedFVs);
			
			int size = m_testSet.size();
			if (m_cache==null || m_cache.length < size*(size+1)/2)
				m_cache = new double[size*(size+1)/2];
			
			_Doc di, dj;
			for(int i=1; i<size; i++) {
				di = m_testSet.get(i);
				for(int j=0; j<i; j++) {
					dj = m_testSet.get(j);
					double similarity = getSimilarity(di, dj);

					m_cache[getIndex(i, j)-1] = similarity;
					if (Math.random() > m_contSamplingRate)
						continue;
					
					//plot as binary categories
					writer.write(String.format("%s %.4f\n", di.getYLabel()==dj.getYLabel(), similarity));					
					//plot all categories
//					if (di.getYLabel()<dj.getYLabel())
//						writer.write(String.format("%s-%s %.4f\n", di.getYLabel(), dj.getYLabel(), similarity));
//					else
//						writer.write(String.format("%s-%s %.4f\n", dj.getYLabel(), di.getYLabel(), similarity));
				}
				
//				for(int j=0; j<m_trainSet.size(); j++) {
//					if (Math.random() > m_contSamplingRate)
//						continue;
//					
//					dj = m_trainSet.get(j);					
//
//					String name = (new Boolean(di.getYLabel()==dj.getYLabel())).toString();
//					double similarity = getSimilarity(di, dj);
//					//plot as binary categories
//					//writer.write(String.format("%s %.4f\n", name, similarity));					
//					//plot all categories
//					if (di.getYLabel()<dj.getYLabel())
//						writer.write(String.format("%s-%s %.4f\n", di.getYLabel(), dj.getYLabel(), similarity));
//					else
//						writer.write(String.format("%s-%s %.4f\n", dj.getYLabel(), di.getYLabel(), similarity));
//					
//					ranklist.add(new _RankItem(name, similarity));
//				}
			}
			writer.close();
			
			MyPriorityQueue<_RankItem> ranklist = new MyPriorityQueue<_RankItem>(50);
			double prec = 0;
			int label;
			for(int i=0; i<size; i++) {
				di = m_testSet.get(i);
				label = di.getYLabel();
				
				for(int j=0; j<size; j++) {
					if (i==j)
						continue;
					dj = m_testSet.get(j);
					ranklist.add(new _RankItem(dj.getYLabel(), m_cache[getIndex(i,j)-1]));
				}
			
				double p = 0;
				for(_RankItem it:ranklist) {
					if (it.m_index == label)
						p ++;
				}
				
				prec += p/ranklist.size();
				ranklist.clear();
			}
			
			System.out.format("Similarity ranking P@%d: %.3f\n", ranklist.size(), prec/size);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	//temporary code for verifying similarity calculation
	public void verification(int k, _Corpus c, String filename){
		c.shuffle(k);
		int[] masks = c.getMasks();
		ArrayList<_Doc> docs = c.getCollection();
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < masks.length; j++) {
				//more for testing
//						if( masks[j]==(i+1)%k || masks[j]==(i+2)%k || masks[j]==(i+3)%k ) 
//							m_testSet.add(docs.get(j));
//						else if (masks[j]==i)
//							m_trainSet.add(docs.get(j));
				
				//more for training
				if(masks[j]==i) 
					m_testSet.add(docs.get(j));
				else
					m_trainSet.add(docs.get(j));
			}
			
			init();//no need to call train()
			verifySimilarity(filename, i!=0);
			
			m_trainSet.clear();
			m_testSet.clear();
		}
	}
}