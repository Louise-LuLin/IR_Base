package Classifier.semisupervised.CoLinAdapt;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;

import structures._Doc;
import structures._Review;
import structures._Review.rType;
import structures._User;

public class _LinAdaptStruct {
	int m_id = 0; // by default all users have the same user ID
	protected double[] m_A; // transformation matrix which is 2*(k+1) dimension.
	protected _User m_user; // unit to store train/adaptation/test data
	protected int m_dim; // number of feature groups
	protected LinkedList<_Review> m_adaptCache;// adaptation cache to hold recent observations, default size is one
	protected int m_cacheSize = 1;
	private int m_adaptPtr, m_adaptStartPos, m_adaptEndPos;		
	
	public _LinAdaptStruct(_User user, int dim) {
		m_user = user;
		
		m_dim = dim;
		m_A = new double[dim*2];		
		for(int i=0; i < m_dim; i++)
			m_A[i] = 1;//Scaling in the first dim dimensions. Initialize scaling to be 1 and shifting be 0.
		
		//trace to the first adaptation instance
		ArrayList<_Review> reviews = user.getReviews();
		for(int i=0; i<reviews.size(); i++) {
			if (reviews.get(m_adaptStartPos).getType() == rType.ADAPTATION)
				m_adaptStartPos = i;
			else if (reviews.get(m_adaptStartPos).getType() == rType.TEST) {
				m_adaptEndPos = i;
				break;
			}
		}
		
		resetAdaptPtr();
	}		
	
	public ArrayList<_Review> getReviews(){
		return m_user.getReviews();
	}
	
	public int getAdaptationSize() {
		return m_adaptEndPos - m_adaptStartPos;
	}
	
	public int getAdaptedCount() {
		return m_adaptPtr - m_adaptStartPos;
	}
	
	// Get one review from a user's reviews.
	public Collection<_Review> nextAdaptationIns(){
		ArrayList<_Review> reviews = m_user.getReviews();
		
		//reach the maximum storage
		if (m_adaptCache.size()>=m_cacheSize)
			m_adaptCache.poll();
		
		if(m_adaptPtr < m_adaptEndPos){
			m_adaptCache.add(reviews.get(m_adaptPtr));
			m_adaptPtr++; //Move to the next review of the current user.
		}
		return m_adaptCache;
	}
	
	public boolean hasNextAdaptationIns() {
		return m_adaptPtr < m_adaptEndPos;
	}
	
	public void resetAdaptPtr() {
		m_adaptPtr = m_adaptStartPos;
		if (m_adaptCache==null) 
			m_adaptCache = new LinkedList<_Review>();
		else
			m_adaptCache.clear();				
	}
	
	public void setPersonalizedModel(double[] pWeight, int classNo, int featureSize) {
		m_user.setModel(pWeight, classNo, featureSize);
	}
	
	public int predict(_Doc doc) {
		return m_user.predict(doc);
	}
	
	public double[] getA() {
		return m_A;
	}
	
	//get the shifting operation for this group
	public double getShifting(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			return Double.NaN;
		}
		return m_A[m_dim+gid];
	}
	
	//get the shifting operation for this group
	public double getScaling(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			return Double.NaN;
		}
		return m_A[gid];
	}
}
