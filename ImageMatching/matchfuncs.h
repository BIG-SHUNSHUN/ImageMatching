#pragma once

#include <opencv2/opencv.hpp>
#include "phase.h"

using namespace cv;

namespace shun
{
	/*
	使用模板匹配的方法匹配单个点

	imgL : 左影像
	imgR ：有影像 
	pts : 要匹配的点
	tempSize : 模板窗口大小，一定是奇数
	method : 相似性测度方法，一般使用带NORM的方法，这样对光照有一定的不变性
			TM_SQDIFF：差平方和测度
			TM_SQDIFF_NORMED：归一化差平方和测度，也就是除以窗口的灰度平方和
			TM_CCORR：相关积测度
			TM_CCORR_NORMED：归一化相关积测度，也就是除以窗口的灰度平方和
			TM_CCOEFF：协方差测度
			TM_CCOEFF_NORMED：归一化协方差测度，也就是相关系数啦
	*/
	Point TemplateMatchingForPoint(Mat imgL, Mat imgR, const Point& pt, int tempSize, TemplateMatchModes method);

	Mat MatchingUsingSIFT(Mat left, Mat right);



	Mat FastSampleConsensus(std::vector<Point2f> small1, 
		                    std::vector<Point2f> small2,
		                    std::vector<Point2f> large1,
		                    std::vector<Point2f> large2,
		                    int iters);
}