#include "features.h"

void shun::HarrisDetector::DetectAndCompute(const Mat & img, std::vector<Point>& pts)
{
	Mat harris;
	cornerHarris(img, harris, _blockSize, _kernelSize, _k, BORDER_CONSTANT);

	//Mat normalized;
	//normalize(harris, normalized, 0, 255, NORM_MINMAX, CV_8U);
	//double minVal, maxVal;
	//minMaxLoc(normalized, &minVal, &maxVal);
	//uchar thresh = saturate_cast<uchar>((1 - _threshRatio) * maxVal + _threshRatio * minVal);
	//Mat betterThanThresh = normalized > thresh;
	
	// 极大值抑制，窗口为5 * 5
	Mat dilated;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(harris, dilated, kernel);
	Mat nonMaxSuppression = dilated == harris;

	//Mat result;
	//bitwise_and(betterThanThresh, nonMaxSuppression, result);

	int rows = nonMaxSuppression.rows;
	int cols = nonMaxSuppression.cols;
	for (int i = 0; i < rows; i++)
	{
		uchar* ptr = nonMaxSuppression.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			if (ptr[j])
				pts.push_back(Point(j, i));
		}
	}
}

void shun::DrawFeaturePoints(Mat img, const std::vector<Point>& pts)
{
	assert(img.type() == CV_8UC3);
	for (int i = 0; i < pts.size(); i++)
	{
		circle(img, pts[i], 3, Scalar(0, 0, 255), -1);
	}
}

void shun::ShiTomashiDetector::DetectAndCompute(const Mat & img, std::vector<Point>& pts)
{
	goodFeaturesToTrack(img, pts, _maxCorners, _qualityLevel, _minDistance, Mat(), _blockSize, _useHarris, _k);
}
