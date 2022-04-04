#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "phase.h"

namespace shun
{
	/* Harrisj角点算子

	   1）经过non-maximum suppression后的点太多了
	   2）如果通过设定一个阈值_threshold筛选点，留下的点分布不均匀
	   3）参数要根据影像的不同而调整
	*/
	class HarrisDetector
	{
	public:
		HarrisDetector(int blockSize = 7, int kernelSize = 3, double k = 0.04)
			: _blockSize(blockSize), _kernelSize(kernelSize), _k(k), _threshRatio(0.5) {}
		
		/* 提取
			
		@params
			img：输入图像
			result：提取到的特征点，以push_back的方式
			radius：以特征点为中心，半径为radius的窗口超出了图像范围，则拒绝这个特征点，
			        为模板匹配算法提供方便
		*/
		int DetectAndCompute(const cv::Mat& img, std::vector<cv::Point2f>& result, int radius);

	private:
		int _blockSize;
		int _kernelSize;
		double _k;
		double _threshRatio;
	};

	/* Shi-tomashi角点算子
	
	   这个检测算子有以下几个特征：
	   1）以自相关矩阵的最小特征值或者harris响应值作为quality value，由参数useHarris控制
	   2）算法内部已经做了3 * 3的非极大值抑制
	   3）quality value 小于 qualityLevel * [最大quality value]的角点被rejected
	   4）剩下的角点根据quality value进行排序
	   5）某个角点的邻域内，如果还有一个quality value更大的点，这个角点应该丢弃，邻域大小由minDistance控制，
	      这可以让角点的分布均匀一些（这是在非极大值抑制后做的）
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

	/* 非线性尺度空间
	
	参考源码：https://github.com/yyxgiser/HAPCG-Multimodal-matching
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
			nLayer：层数
			scaleValue：层间缩放倍数
			_whichDiff：使用的扩散函数
			sigma1：第一层尺度
			sigma2：第二层以上的高斯平滑标准差
			ratio：层间的尺度比
			perc：分位数，一般不变
		*/
		NonlinearSpace(int nLayer = 3, double scaleValue = 1.6, DIFFUSION_FUNCTION _whichDiff = G2, 
			           double sigma1 = 1.6, double sigma2 = 1, double ratio = pow(2, 1.0 / 3), double perc = 0.7);
		
		void Generate(cv::Mat imgIn);
		cv::Mat GetLayer(int i);
		int size() { return _space.size(); }
		double zoom() { return _scaleValue; }

	private:
		
		// 参数
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

	/* TODO 残次品，有待加工

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

	/* RIFT特征检测与描述

	refrence paper：https://ieeexplore.ieee.org/document/8935498

	实际效果不错，值得研究
	*/
	class RIFT
	{
	public:

		/*
		@params
			nScale：log-gabor尺度数
			nOrient：log-gabor方向数
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

	/* 方向断面检测法

	@params
		g_map：梯度图
		d_map：方向图
		mask：掩膜，如果mask.at<uchar>(r, c) == 0，那么点(r, c) 所在的像素不作为边缘点的候选点 
	*/
	cv::Mat OrientCrossDetection(cv::Mat g_map, cv::Mat d_map, cv::Mat mask = cv::Mat());
}