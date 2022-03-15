#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

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
		
		void DetectAndCompute(const cv::Mat& img, std::vector<cv::Point>& result);

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
	   5��ĳ���ǵ�������ڣ�����һ��quality value����ĵ㣬����ǵ�Ӧ�ö����������С��minDistance���ƣ�
	      ������ýǵ�ķֲ�����һЩ�������ڷǼ���ֵ���ƺ����ģ�
	*/
	class ShiTomashiDetector
	{
	public:
		ShiTomashiDetector(int maxCorners, bool useHarris = false, double qualityLevel = 0.01, double minDistance = 10,
			               int blockSize = 5, double k = 0.04)
			: _maxCorners(maxCorners), _qualityLevel(qualityLevel), _minDistance(minDistance),
			  _blockSize(blockSize), _useHarris(useHarris), _k(k) {}

		void DetectAndCompute(const cv::Mat& img, std::vector<cv::Point>& pts);

	private:
		int _maxCorners;
		double _qualityLevel;
		double _minDistance;
		int _blockSize;
		bool _useHarris;
		double _k;

	};

	void DrawFeaturePoints(cv::Mat img, const std::vector<cv::Point>& pts);

	class NonlinearSpace
	{
		enum DIFFUSION_FUNCTION
		{
			G1,
			G2,
			G3
		};
	public:
		NonlinearSpace(int nLayer = 3, double scaleValue = 1.6, DIFFUSION_FUNCTION _whichDiff = G2, 
			           double sigma1 = 1.6, double sigma2 = 1, double ratio = pow(2, 1.0 / 3), double perc = 0.7);
		void Generate(cv::Mat imgIn);
		cv::Mat GetLayer(int i);

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
}
