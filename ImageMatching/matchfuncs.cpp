#include "matchfuncs.h"
#include <vector>
#include <random>

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

Mat shun::FastSampleConsensus(std::vector<Point2f> small1, std::vector<Point2f> small2, std::vector<Point2f> large1, std::vector<Point2f> large2, int iters)
{
	default_random_engine e;
	e.seed(time(NULL));

	vector<Point2f> randPts1(4), randPts2(4);
	vector<Point2f> transformed;
	Mat transformedMatrix;
	int n = small1.size();
	int maxCount = 0;
	for (int i = 0; i < iters; i++)
	{
		uniform_int_distribution<int> randGenerator(0, n - 1);
		for (int i = 0; i < 4; i++)
		{
			int index = randGenerator(e);
			randPts1[i] = small1[index];
			randPts2[2] = small2[index];
		}

		Mat m =	findHomography(randPts1, randPts2);

		perspectiveTransform(large1, transformed, m);

		int count = 0;
		for (int i = 0; i < transformed.size(); i++)
		{
			float x1 = transformed[i].x;
			float y1 = transformed[i].y;
			float x2 = large2[i].x;
			float y2 = large2[i].y;
			float dx = x2 - x1;
			float dy = y2 - y1;

			if (sqrt(dx * dx + dy * dy) < 1.0)
			{
				count++;
			}
		}

		if (count > maxCount)
		{
			maxCount = count;
			transformedMatrix = m;
		}
	}

	return transformedMatrix;
}


