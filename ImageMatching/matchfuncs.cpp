#include "matchfuncs.h"
#include <vector>

using namespace std;

Point shun::TemplateMatchingForPoint(Mat imgL, Mat imgR, const Point & pt, int tempSize, TemplateMatchModes method)
{
	int x = pt.x - tempSize / 2;
	int y = pt.y - tempSize / 2;
	Mat templ = imgL(Rect(x, y, tempSize, tempSize));

	Mat result;
	matchTemplate(imgR, templ, result, method);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	Point matchLoc;
	if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
		matchLoc = minLoc;
	else
		matchLoc = maxLoc;

	return Point(matchLoc.x + tempSize / 2, matchLoc.y + tempSize / 2);
}

Mat shun::MatchingUsingSIFT(Mat left, Mat right)
{
	imshow("left", left);
	imshow("right", right);
	// 探测特征点
	vector<KeyPoint> keyPointLeft, keyPointRight;
	Ptr<SiftFeatureDetector> f2d = SIFT::create(0, 3, 0.04, 10, 1.6);
	f2d->detect(left, keyPointLeft);
	f2d->detect(right, keyPointRight);
	// 计算描述向量
	Mat descriptorLeft, descriptorRight;
	f2d->compute(left, keyPointLeft, descriptorLeft);
	f2d->compute(right, keyPointRight, descriptorRight);
	//匹配特征点
	FlannBasedMatcher matcher;
	vector<DMatch> matchResult;
	matcher.match(descriptorLeft, descriptorRight, matchResult);
	// 画出匹配图像
	Mat initialMatches;
	drawMatches(left, keyPointLeft, right, keyPointRight, matchResult, initialMatches);
	imshow("最初匹配", initialMatches);
	waitKey(0);
	// 寻找好的匹配点
	float minDist = matchResult[0].distance, maxDist = matchResult[0].distance;
	for (int i = 1; i < matchResult.size(); i++)
	{
		minDist = matchResult[i].distance < minDist ? matchResult[i].distance : minDist;
		maxDist = matchResult[i].distance > maxDist ? matchResult[i].distance : maxDist;
	}
	vector<DMatch> goodMatchResult;
	for (int i = 0; i < matchResult.size(); i++)
	{
		if (matchResult[i].distance <= 2 * minDist)
			goodMatchResult.push_back(matchResult[i]);
	}
	// 画出匹配图像（好的匹配点）
	Mat goodMatches;
	drawMatches(left, keyPointLeft, right, keyPointRight, goodMatchResult, goodMatches);
	imshow("好的匹配", goodMatches);
	waitKey(0);
	// 计算变换矩阵
	vector<Point2f> pointLeft, pointRight;
	for (int i = 0; i < goodMatchResult.size(); i++)
	{
		pointLeft.push_back(keyPointLeft[goodMatchResult[i].queryIdx].pt);
		pointRight.push_back(keyPointRight[goodMatchResult[i].trainIdx].pt);
	}
	Mat transformMat = findHomography(pointRight, pointLeft, RANSAC);
	// 影像配准
	vector<Point2f> rightCornerPoint(4), rightTransformedPoints(4);
	rightCornerPoint[0].x = 0; rightCornerPoint[0].y = 0;
	rightCornerPoint[1].x = 0; rightCornerPoint[1].y = right.rows;
	rightCornerPoint[2].x = right.cols; rightCornerPoint[2].y = right.rows;
	rightCornerPoint[3].x = right.cols; rightCornerPoint[3].y = 0;
	perspectiveTransform(rightCornerPoint, rightTransformedPoints, transformMat);
	// 拼接
	Mat dst;
	warpPerspective(right, dst, transformMat, Size(700, 700));
	left.copyTo(dst(Rect(0, 0, left.cols, left.rows)));
	line(dst, rightTransformedPoints[0], rightTransformedPoints[1], Scalar(255, 255, 255));
	line(dst, rightTransformedPoints[1], rightTransformedPoints[2], Scalar(255, 255, 255));
	line(dst, rightTransformedPoints[2], rightTransformedPoints[3], Scalar(255, 255, 255));
	line(dst, rightTransformedPoints[3], rightTransformedPoints[0], Scalar(255, 255, 255));
	imshow("拼接结果", dst);
	waitKey(0);
	return dst;
}

bool CompareFunc(const KeyPoint& lhs, const KeyPoint& rhs)
{
	return lhs.response > rhs.response;
}

shun::RIFT::RIFT(int nScale, int nOrient)
	: _pc(nScale, nOrient)
{
}

shun::RIFT::RIFT(const PhaseCongruency & pc)
	: _pc(pc)
{
}

void shun::RIFT::DetectAndCompute(Mat imgIn, vector<KeyPoint>& keyPoints, Mat& descriptors)
{
	DetectFeature(imgIn, keyPoints);

	int nOrient = _pc._nOrient;
	int nScale = _pc._nScale;
	EO eo = _pc._eo;
	vector<Mat> CS(nOrient);
	for (int o = 0; o < nOrient; o++)
	{
		Mat tmp = Mat::zeros(imgIn.size(), CV_64FC1);
		for (int s = 0; s < nScale; s++)
		{
			Mat matArr[2];
			split(eo[s][o], matArr);

			Mat mag;
			magnitude(matArr[0], matArr[1], mag);
			tmp = tmp + mag;
		}
		CS[o] = tmp;
	}
	Mat MIM = BuildMIM(CS);

	descriptors.create(_ns * _ns * nOrient, keyPoints.size(), CV_64FC1);
	for (int i = 0; i < keyPoints.size(); i++)
	{
		int x = keyPoints[i].pt.x;
		int y = keyPoints[i].pt.y;

		int x1 = x - _patchSize / 2;
		int y1 = y - _patchSize / 2;
		int x2 = x + _patchSize / 2;
		int y2 = y + _patchSize / 2;

		Mat patch(MIM, Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
		int ys = patch.rows;
		int xs = patch.cols;
		Mat RIFT_des = Mat::zeros(_ns * _ns * nOrient, 1, CV_64FC1);
		for (int j = 0; j < _ns; j++)
		{
			for (int k = 0; k < _ns; k++)
			{
				double step = (double)ys / _ns;
				int yc1 = round(j * step);
				int yc2 = round((j + 1) * step);
				int xc1 = round(k * step);
				int xc2 = round((k + 1) * step);

				Mat clip(patch, Rect(xc1, yc1, xc2 - xc1, yc2 - yc1));

				Mat hist;
				float ranges[] = { 1, _ns + 1 };
				const float* histRange = { ranges };
				calcHist(&clip, 1, 0, Mat(), hist, 1, &_ns, &histRange);

				hist.convertTo(hist, CV_64FC1);
				Mat roi(RIFT_des, Rect(0, nOrient * (j * _ns + k), 1, nOrient));
				hist.copyTo(roi);
			}
		}

		double normVal = norm(RIFT_des, NORM_L2);
		if (normVal != 0)
			RIFT_des = RIFT_des / normVal;

		RIFT_des.copyTo(descriptors.col(i));
	}
}

bool PtsCompare(const KeyPoint& lhs, const KeyPoint& rhs)
{
	return lhs.response > rhs.response;
}

void shun::RIFT::DetectFeature(Mat imgIn, vector<KeyPoint>& keyPoints)
{
	_pc.Calc(imgIn);

	Mat M, m;
	_pc.Feature(M, m);

	normalize(M, M, 0, 255, NORM_MINMAX);
	M.convertTo(M, CV_8UC1);

	vector<KeyPoint> pts;
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
	detector->detect(M, pts);

	sort(pts.begin(), pts.end(), PtsCompare);

	for (int i = 0; i < pts.size() && i < _ptsNum; i++)
	{
		int x = pts[i].pt.x;
		int y = pts[i].pt.y;

		int x1 = x - _patchSize / 2;
		int y1 = y - _patchSize / 2;
		int x2 = x + _patchSize / 2;
		int y2 = y + _patchSize / 2;

		if (x1 < 0 || y1 < 0 || x2 >= imgIn.cols || y2 >= imgIn.rows)
		{
			continue;
		}
		else
		{
			keyPoints.push_back(pts[i]);
		}
	}
}

Mat shun::RIFT::BuildMIM(vector<Mat>& CS)
{
	int rows = CS[0].rows;
	int cols = CS[0].cols;

	Mat MIM = Mat::zeros(rows, cols, CV_8UC1);
	for (int r = 0; r < rows; r++)
	{
		uchar* ptrMIM = MIM.ptr<uchar>(r);
		for (int c = 0; c < cols; c++)
		{
			int iMax = 0;
			double maxVal = CS[0].at<double>(r, c);
			for (int o = 1; o < CS.size(); o++)
			{
				double val = CS[o].at<double>(r, c);
				if (val > maxVal)
				{
					iMax = o;
					maxVal = val;
				}
			}
			ptrMIM[c] = iMax + 1;
		}
	}

	return MIM;
}
