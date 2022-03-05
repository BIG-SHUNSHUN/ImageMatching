#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

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
		
		void DetectAndCompute(const Mat& img, std::vector<Point>& result);

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
	   5）某个角点的邻域内，还有一个quality value更大的点，这个角点应该丢弃，邻域大小由minDistance控制，
	      这可以让角点的分布均匀一些（这是在非极大值抑制后做的）
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
