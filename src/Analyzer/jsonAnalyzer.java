package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.Post;
import structures.Product;
import structures._Doc;
import utils.Utils;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class jsonAnalyzer extends DocAnalyzer{
	
	SimpleDateFormat m_dateFormatter;
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
//		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
		m_dateFormatter = new SimpleDateFormat("yyyy-MM-dddd");// standard date format for this project

	}
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
//		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
		m_dateFormatter = new SimpleDateFormat("yyyy-MM-dddd");// standard date format for this project
	}
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel, String tagModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold, tagModel);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
	}
	//previous LoadDoc, in case we need it.
	public void LoadDoc(String filename) {
		Product prod = null;
		JSONArray jarray = null;
		
		try {
			JSONObject json = LoadJson(filename);
			//prod = new Product(json.getJSONObject("ProductInfo"));
			jarray = json.getJSONArray("Reviews");
		} catch (Exception e) {
			System.out.print('X');
			return;
		}	
		
		for(int i=0; i<jarray.length(); i++) {
			try {
				Post post = new Post(jarray.getJSONObject(i));
				if (checkPostFormat(post)){
					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
					_Doc review = new _Doc(m_corpus.getSize(), post.getContent(), post.getLabel()-1, timeStamp);
					if(this.m_stnDetector!=null){
						AnalyzeDocWithStnSplit(review);
					}
					else
						AnalyzeDoc(review);
				}
			} catch (ParseException e) {
				System.out.print('T');
			} catch (JSONException e) {
				System.out.print('P');
			}
		}
	}
	
//	public void LoadDoc(String filename) {
//		Product prod = null;
//		JSONArray jarray = null;
//		
//		try {
//			JSONObject json = LoadJson(filename);
//			//prod = new Product(json.getJSONObject("ProductInfo"));
//			jarray = json.getJSONArray("Reviews");
//		} catch (Exception e) {
//			System.out.print('X');
//			return;
//		}	
//		
//		for(int i=0; i<jarray.length(); i++) {
//			try {
//				Post post = new Post(jarray.getJSONObject(i));
//				if (checkPostFormat(post)){
//					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
//					String content;
//					if (Utils.endWithPunct(post.getTitle()))
//						content = post.getTitle() + " " + post.getContent();
//					else
//						content = post.getTitle() + ". " + post.getContent();
////					int label = 0;
////					if(post.getLabel()>=4) label = 1;
////					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), post.getTitle(), prod.getID(), label, timeStamp);
//					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), post.getTitle(), content, prod.getID(), post.getLabel()-1, timeStamp);
//					if(this.m_stnDetector!=null){
//						AnalyzeDocWithStnSplit(review);
//					}
//					else
//						AnalyzeDoc(review);
//				}
//			} catch (ParseException e) {
//				System.out.print('T');
//			} catch (JSONException e) {
//				System.out.print('P');
//			}
//		}
//	}
	/***Load a json file with 1:1 postive and negative reviews.
	Will this ruin the bias given by the ratio since this ratio is informative itself.***/
//	public void LoadDoc(String filename) {
//		Product prod = null;
//		JSONArray jarray = null;
//		int[] count = new int[m_classNo];
//		
//		try {
//			JSONObject json = LoadJson(filename);
//			prod = new Product(json.getJSONObject("ProductInfo"));
//			jarray = json.getJSONArray("Reviews");
//		} catch (Exception e) {
//			System.out.print('X');
//			return;
//		}	
//		//Get the minimum number of reviews
//		for(int i = 0; i < jarray.length(); i++){
//			try {
//				Post post = new Post(jarray.getJSONObject(i));
//				if (checkPostFormat(post)){
//					int label = 0;
//					if(post.getLabel() >= 3) label = 1;
//					count[label]++;
//				}
//			} catch (JSONException e) {
//				System.out.print('P');
//			}
//		}
//		int min = Utils.minOfArrayValue(count);
//		count = new int[m_classNo];
//		for(int i=0; i<jarray.length(); i++) {
//			try {
//				Post post = new Post(jarray.getJSONObject(i));
//				if (checkPostFormat(post)){
//					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
//					String content;
//					if (Utils.endWithPunct(post.getTitle()))
//						content = post.getTitle() + " " + post.getContent();
//					else
//						content = post.getTitle() + ". " + post.getContent();
//
//					int label = 0;
//					if(post.getLabel() >= 3) label = 1;
//					if(count[label] < min){
//						_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), label, timeStamp);
//						count[label]++;
////						_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), post.getLabel()-1, timeStamp);
//						if(this.m_stnDetector!=null)
//							AnalyzeDocWithStnSplit(review);
//						else
//							AnalyzeDoc(review);
//					}
//				}
//			} catch (ParseException e) {
//				System.out.print('T');
//			} catch (JSONException e) {
//				System.out.print('P');
//			}
//		}
//	}
	
	//sample code for loading the json file
	JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			return new JSONObject(buffer.toString());
		} catch (Exception e) {
			System.out.print('X');
			return null;
		}
	}
	
	//check format for each post
	protected boolean checkPostFormat(Post p) {
		if (p.getLabel() <= 0 || p.getLabel() > 5){
			//System.err.format("[Error]Missing Lable or wrong label!!");
			System.out.print('L');
			return false;
		}
		else if (p.getContent() == null){
			//System.err.format("[Error]Missing content!!\n");
			System.out.print('C');
			return false;
		}	
		else if (p.getDate() == null){
			//System.err.format("[Error]Missing date!!\n");
			System.out.print('d');
			return false;
		}
		else {
			// to check if the date format is correct
			try {
				m_dateFormatter.parse(p.getDate());
				//System.out.println(p.getDate());
				return true;
			} catch (ParseException e) {
				System.out.print('D');
			}
			return true;
		} 
	}
}
