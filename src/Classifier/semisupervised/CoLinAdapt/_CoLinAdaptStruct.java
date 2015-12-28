/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Collection;
import java.util.LinkedList;

import structures.MyPriorityQueue;
import structures._RankItem;
import structures._User;

/**
 * @author Hongning Wang
 * Add the shared structure for efficiency purpose
 */
public class _CoLinAdaptStruct extends _LinAdaptStruct {
	public enum SimType {
		ST_BoW,
		ST_SVD
	}
	
	static double[] sharedA;	
	MyPriorityQueue<_RankItem> m_neighbors; //top-K neighborhood, we only store an asymmetric graph structure
	LinkedList<_RankItem> m_reverseNeighbors; // this user contributes to the other users' neighborhood
	
	public _CoLinAdaptStruct(_User user, int dim, int id, int topK) {
		super(user, dim);
		m_id = id;
		m_neighbors = new MyPriorityQueue<_RankItem>(topK);
		m_reverseNeighbors = new LinkedList<_RankItem>();
	}
	
	public void addNeighbor(int id, double similarity) {
		m_neighbors.add(new _RankItem(id, similarity));
	}
	
	public void addReverseNeighbor(int id, double similarity) {
		for(_RankItem it:m_neighbors) {
			if (it.m_index == id)
				return;
		}
		
		m_reverseNeighbors.add(new _RankItem(id, similarity));
	}
	
	public double getSimilarity(_CoLinAdaptStruct user, SimType sType) {
		if (sType == SimType.ST_BoW)
			return user.m_user.getBoWSim(m_user);
		else
			return user.m_user.getSVDSim(m_user);
	}
	
	public Collection<_RankItem> getNeighbors() {
		return m_neighbors;
	}
	
	public Collection<_RankItem> getReverseNeighbors() {
		return m_reverseNeighbors;
	}

	
	static public double[] getSharedA() {
		return sharedA;
	}
	
	//this operation becomes very expensive in _CoLinStruct
	@Override
	public double[] getA() {
		int offset = m_id * m_dim * 2;
		System.arraycopy(sharedA, offset, m_A, 0, m_dim*2);
		return m_A;
	}
	
	//get the shifting operation for this group
	@Override
	public double getShifting(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			return Double.NaN;
		}
		
		int offset = m_id * m_dim * 2;
		return sharedA[offset+m_dim+gid];
	}
	
	public void setShifting(int gid, double value) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			System.exit(-1);
		}
		
		int offset = m_id * m_dim * 2;
		sharedA[offset+m_dim+gid] = value;
	}
	
	//get the shifting operation for this group
	@Override
	public double getScaling(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			return Double.NaN;
		}
		
		int offset = m_id * m_dim * 2;
		return sharedA[offset+gid];
	}
	
	public void setScaling(int gid, double value) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			System.exit(-1);
		}
		
		int offset = m_id * m_dim * 2;
		sharedA[offset+gid] = value;
	}
}
