#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "phase.h"

namespace shun
{
	/* Harrisj�ǵ�����

	   1������non-maximum suppression��ĵ�̫����
	   2�����ͨ���趨һ����ֵ_thresholdɸѡ�㣬���µĵ�ֲ�������
	   3������Ҫ����Ӱ��Ĳ�ͬ������
	*/
	class HarrisDetector
	{
	public:
		HarrisDetector(int blockSize = 7, int kernelSize = 3, double k = 0.04)
			: _blockSize(blockSize), _kernelSize(kernelSize), _k(k), _threshRatio(0.5) {}
		
		/* ��ȡ
			
		@params
			img������ͼ��
			result����ȡ���������㣬��push_back�ķ�ʽ
			radius����������Ϊ���ģ��뾶Ϊradius�Ĵ��ڳ�����ͼ��Χ����ܾ���������㣬
			        Ϊģ��ƥ���㷨�ṩ����
		*/
		int DetectAndCompute(const cv::Mat& img, std::vector<cv::Point2f>& result, int radius);

	private:
		int _blockSize;
		int _kernelSize;
		double _k;
		double _threshRatio;
	};

	/* Shi-tomashi�ǵ�����
	
	   ���������������¼���������
	   1��������ؾ������С����ֵ����harris��Ӧֵ��Ϊquality value���ɲ���useHarris����
	   2���㷨�ڲ��Ѿ�����3 * 3�ķǼ���ֵ����
	   3��quality value С�� qualityLevel * [���quality value]�Ľǵ㱻rejected
	   4��ʣ�µĽǵ����quality value��������
	   5��ĳ���ǵ�������ڣ��������һ��quality value����ĵ㣬����ǵ�Ӧ�ö����������С��minDistance���ƣ�
	      ������ýǵ�ķֲ�����һЩ�������ڷǼ���ֵ���ƺ����ģ�
	*/
	class ShiTomashiDetector
	{
	public:
		ShiTomashiDetector(int maxCorners, bool useHarris = false, 
			               double qualityLevel = 0.1, 
			               double minDistance = 10,
			               int blockSize = 5, double k = 0.04)
			: _maxCorners(maxCorners), _qualityLevel(qualityLevel), _minDistance(minDistance),
			  _blockSize(blockSize), _useHarris(useHarris), _k(k) {}

		void DetectAndCompute(const cv::Mat& img, std::vector<cv::Point2f>& pts, int r = 0);

	private:
		int _maxCorners;
		double _qualityLevel;
		double _minDistance;
		int _blockSize;
		bool _useHarris;
		double _k;

	};

	void DrawFeaturePoints(cv::Mat img, const std::vector<cv::Point2f>& pts);

	/* �����Գ߶ȿռ�
	
	�ο�Դ�룺https://github.com/yyxgiser/HAPCG-Multimodal-matching
	*/
	class NonlinearSpace
	{
		enum DIFFUSION_FUNCTION
		{
			G1,
			G2,
			G3
		};
	public:
		/* 
		@params
			nLayer������
			scaleValue��������ű���
			_whichDiff��ʹ�õ���ɢ����
			sigma1����һ��߶�
			sigma2���ڶ������ϵĸ�˹ƽ����׼��
			ratio�����ĳ߶ȱ�
			perc����λ����һ�㲻��
		*/
		NonlinearSpace(int nLayer = 3, double scaleValue = 1.6, DIFFUSION_FUNCTION _whichDiff = G2, 
			           double sigma1 = 1.6, double sigma2 = 1, double ratio = pow(2, 1.0 / 3), double perc = 0.7);
		
		void Generate(cv::Mat imgIn);
		cv::Mat GetLayer(int i);
		int size() { return _space.size(); }
		double zoom() { return _scaleValue; }

	private:
		
		// ����
		int _nLayer;
		double _scaleValue;
		DIFFUSION_FUNCTION _whichDiff;
		double _sigma1;
		double _sigma2;
		double _ratio;
		double _perc;

		std::vector<cv::Mat> _space;

		cv::Mat PM_G1(cv::Mat lx, cv::Mat ly, double k);
		cv::Mat PM_G2(cv::Mat lx, cv::Mat ly, double k);
		cv::Mat PM_G3(cv::Mat lx, cv::Mat ly, double k);
		double K_PercentileValue(cv::Mat lx, cv::Mat ly, double perc);

		cv::Mat AOS(cv::Mat last, double step, cv::Mat diff);
		cv::Mat AOS_row(cv::Mat last, double step, cv::Mat diff);
		cv::Mat AOS_col(cv::Mat last, double step, cv::Mat diff);
		cv::Mat Thomas_Algorithm(cv::Mat a, cv::Mat b, cv::Mat c, cv::Mat d);
	};

	/* TODO �д�Ʒ���д��ӹ�

	TODO
	
	*/
	class HAPCG
	{
	public:
		HAPCG(int layer = 3, double scaleVal = 2.0);
		~HAPCG();
		void DetectAndCompute(cv::Mat imgIn, std::vector<cv::Point2f>& keypoints, cv::Mat& descriptors, 
			                  std::vector<int>& layerBelongTo);
		double zoom() { return _space.zoom(); }

	private:
		NonlinearSpace _space;
		std::vector<cv::Mat> _W;
		std::vector<cv::Mat> _grad;
		std::vector<cv::Mat> _angle;

		double _deltaPhi;
		std::vector<int> _blockRadius;  // z = sqrt(17) * x, y = 3 * x

		void InformationFromPhaseCongruency();
		void DetectHarrisCorner(std::vector<cv::Point2f>& keypoints, std::vector<int>& layerBelongTo);

		cv::Mat Histgram(cv::Mat grad, cv::Mat angle, cv::Mat polarRadius = cv::Mat(), cv::Mat polarAngle = cv::Mat(),
			         int rLow = 0, int rHigh = 0, float angleLow = 0.0, float angleHigh = 0.0);
	};

	/* RIFT�������������

	refrence paper��https://ieeexplore.ieee.org/document/8935498

	ʵ��Ч������ֵ���о�
	*/
	class RIFT
	{
	public:

		/*
		@params
			nScale��log-gabor�߶���
			nOrient��log-gabor������
		*/
		RIFT(int nScale = 4, int nOrient = 6);
		RIFT(const PhaseCongruency& pc);
		RIFT(const RIFT& other) = delete;
		RIFT& operator=(const RIFT& other) = delete;

		void DetectAndCompute(cv::Mat imgIn, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors);

	private:
		PhaseCongruency _pc;
		int _patchSize = 96;
		int _ns = 6;
		int _ptsNum = 5000;

		void DetectFeature(cv::Mat imgIn, std::vector<cv::KeyPoint>& keyPoints);
		cv::Mat BuildMIM(std::vector<cv::Mat>& CS);
	};

	/* ��������ⷨ

	@params
		g_map���ݶ�ͼ
		d_map������ͼ
		mask����Ĥ�����mask.at<uchar>(r, c) == 0����ô��(r, c) ���ڵ����ز���Ϊ��Ե��ĺ�ѡ�� 
	*/
	cv::Mat OrientCrossDetection(cv::Mat g_map, cv::Mat d_map, cv::Mat mask = cv::Mat());
}