#pragma once

#include <opencv2/opencv.hpp>
#include "wavelet.h"

namespace shun
{
	enum TEMPLATE_MATCHING_MODE
	{
		TM_SQDIFF = 0,
		TM_SQDIFF_NORMED = 1,
		TM_CCORR = 2,
		TM_CCORR_NORMED = 3,
		TM_CCOEFF = 4,
		TM_CCOEFF_NORMED = 5,
		MI = 6
	};

	/*
	使用模板匹配的方法匹配单个点

	@params
		imgL : 左影像
		imgR ：右影像 
		pts : 要匹配的点
		tempSize : 模板窗口大小，一定是奇数
		method : 相似性测度方法，一般使用带NORM的方法，这样对光照有一定的不变性
				TM_SQDIFF：差平方和测度
				TM_SQDIFF_NORMED：归一化差平方和测度，也就是除以窗口的灰度平方和
				TM_CCORR：相关积测度
				TM_CCORR_NORMED：归一化相关积测度，也就是除以窗口的灰度平方和
				TM_CCOEFF：协方差测度
				TM_CCOEFF_NORMED：归一化协方差测度，也就是相关系数啦

	@return
		maxSimilarity：最大测度值
		maxLoc：最大测度值所在的位置
	*/
	cv::Point2f TemplateMatchingForPoint(cv::Mat patchL, cv::Mat searchRegion, TEMPLATE_MATCHING_MODE method, 
		                                 double& maxSimilarity);

	cv::Mat MatchingUsingSIFT(cv::Mat left, cv::Mat right);



	cv::Mat FastSampleConsensus(std::vector<cv::Point2f> small1,
		                    std::vector<cv::Point2f> small2,
		                    std::vector<cv::Point2f> large1,
		                    std::vector<cv::Point2f> large2,
		                    int iters);

	class FeatureMatching
	{
	public:
		FeatureMatching();
		~FeatureMatching();

	private:


	};

	void RIFT_Matching(std::string fileL, std::string fileR);

	class WaveletPyramidMathing
	{
	public:
		WaveletPyramidMathing(int patchSize, int searchSize, int nLayer = 3, string name = "haar");
		~WaveletPyramidMathing() {}

		void Execute(std::vector<cv::Point2f>& pts, cv::Mat sensed, cv::Mat reference,
			         std::vector<cv::Point2f>& out);

	private:
		WaveletPyramid _pydSensed;
		WaveletPyramid _pydRef;

		int _nLayer;
		std::string _name;
		int _patchSize;
		int _searchSize;

		cv::Point2f CorePointMatching(cv::Mat sensed, cv::Mat reference);
	};

	/* 模板匹配框架

	在已经基本纠正过的情况下，sensed image和reference image仅有几十个像素的偏差，因此可在reference image
	的小范围内确定一个搜索区，搜索最佳匹配点

	对于重叠度比较大的影像对（大于60%），可以采用核心点匹配。先匹配一个点，根据匹配结果计算出偏移量，匹配时进行
	全图搜索。匹配其他点时根据计算出的偏移量即可指定一个小范围的搜索区。

	对于模板匹配来说，如果图像有重复纹理，那就GG了
	*/
	class TemplateMatching
	{
	public:
		TemplateMatching(int patchSize, int searchSize, TEMPLATE_MATCHING_MODE mode = TM_CCOEFF_NORMED);
		~TemplateMatching() {}

		void Execute(std::vector<cv::Point2f>& pts, cv::Mat sensed, cv::Mat reference,
			         std::vector<cv::Point2f>& out);

	private:
		int _patchSize;     // 模板窗口大小，奇数
		int _searchSize;    // 搜索区，奇数
		TEMPLATE_MATCHING_MODE _mode;

		// similarity metric
		std::string _modeName[7] = { "TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR",
			                   "TM_CCORR_NORMED", "TM_CCOEFF", "TM_CCOEFF_NORMED", "MI"};
	};

	// 计算两个相同的patch之间的互信息（mutual information）
	double MutialInfo(cv::Mat patch1, cv::Mat patch2);

	// 根据直方图计算信息熵
	double Entropy(cv::Mat hist);
}