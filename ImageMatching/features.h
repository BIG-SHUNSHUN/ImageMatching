#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

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
		
		void DetectAndCompute(const Mat& img, std::vector<Point>& result);

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

		void DetectAndCompute(const Mat& img, std::vector<Point>& pts);

	private:
		int _maxCorners;
		double _qualityLevel;
		double _minDistance;
		int _blockSize;
		bool _useHarris;
		double _k;

	};

	void DrawFeaturePoints(Mat img, const std::vector<Point>& pts);
}
