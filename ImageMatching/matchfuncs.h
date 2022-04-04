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
	ʹ��ģ��ƥ��ķ���ƥ�䵥����

	@params
		imgL : ��Ӱ��
		imgR ����Ӱ�� 
		pts : Ҫƥ��ĵ�
		tempSize : ģ�崰�ڴ�С��һ��������
		method : �����Բ�ȷ�����һ��ʹ�ô�NORM�ķ����������Թ�����һ���Ĳ�����
				TM_SQDIFF����ƽ���Ͳ��
				TM_SQDIFF_NORMED����һ����ƽ���Ͳ�ȣ�Ҳ���ǳ��Դ��ڵĻҶ�ƽ����
				TM_CCORR����ػ����
				TM_CCORR_NORMED����һ����ػ���ȣ�Ҳ���ǳ��Դ��ڵĻҶ�ƽ����
				TM_CCOEFF��Э������
				TM_CCOEFF_NORMED����һ��Э�����ȣ�Ҳ�������ϵ����

	@return
		maxSimilarity�������ֵ
		maxLoc�������ֵ���ڵ�λ��
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

	/* ģ��ƥ����

	���Ѿ�����������������£�sensed image��reference image���м�ʮ�����ص�ƫ���˿���reference image
	��С��Χ��ȷ��һ�����������������ƥ���

	�����ص��ȱȽϴ��Ӱ��ԣ�����60%�������Բ��ú��ĵ�ƥ�䡣��ƥ��һ���㣬����ƥ���������ƫ������ƥ��ʱ����
	ȫͼ������ƥ��������ʱ���ݼ������ƫ��������ָ��һ��С��Χ����������

	����ģ��ƥ����˵�����ͼ�����ظ������Ǿ�GG��
	*/
	class TemplateMatching
	{
	public:
		TemplateMatching(int patchSize, int searchSize, TEMPLATE_MATCHING_MODE mode = TM_CCOEFF_NORMED);
		~TemplateMatching() {}

		void Execute(std::vector<cv::Point2f>& pts, cv::Mat sensed, cv::Mat reference,
			         std::vector<cv::Point2f>& out);

	private:
		int _patchSize;     // ģ�崰�ڴ�С������
		int _searchSize;    // ������������
		TEMPLATE_MATCHING_MODE _mode;

		// similarity metric
		std::string _modeName[7] = { "TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR",
			                   "TM_CCORR_NORMED", "TM_CCOEFF", "TM_CCOEFF_NORMED", "MI"};
	};

	// ����������ͬ��patch֮��Ļ���Ϣ��mutual information��
	double MutialInfo(cv::Mat patch1, cv::Mat patch2);

	// ����ֱ��ͼ������Ϣ��
	double Entropy(cv::Mat hist);
}