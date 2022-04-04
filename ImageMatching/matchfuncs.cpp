#include "matchfuncs.h"
#include <vector>
#include <random>
#include "features.h"
#include <fstream>

using namespace std;
using namespace shun;
using namespace cv;

Point2f shun::TemplateMatchingForPoint(Mat patchL, Mat searchRegion, TEMPLATE_MATCHING_MODE method,
	                                   double& maxSimilarity)
{
	int dx = searchRegion.cols - patchL.cols;
	int dy = searchRegion.rows - patchL.rows;
	Mat result = Mat::zeros(dy + 1, dx + 1, CV_32FC1);    // correlation surface
	
	if (method == MI)     // MI
	{
		for (int r = 0; r < dy + 1; r++)
		{
			for (int c = 0; c < dx + 1; c++)
			{
				Mat p(searchRegion, Rect(c, r, patchL.cols, patchL.rows));
				result.ptr<float>(r)[c] = MutialInfo(patchL, p);
			}
		}
	}
	else
	{
		if (patchL.type() == CV_64FC1)
			patchL.convertTo(patchL, CV_32FC1);
		if (searchRegion.type() == CV_64FC1)
			searchRegion.convertTo(searchRegion, CV_32FC1);
		matchTemplate(searchRegion, patchL, result, method);
	}

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);   // �����ֵ����Сֵ

	Point matchLoc;
	if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		maxSimilarity = -minVal;
	}
	else
	{
		matchLoc = maxLoc;
		maxSimilarity = maxVal;
	}
		
	return matchLoc;
}

Mat shun::MatchingUsingSIFT(Mat left, Mat right)
{
	imshow("left", left);
	imshow("right", right);

	// ̽��������
	vector<KeyPoint> keyPointLeft, keyPointRight;
	Ptr<SiftFeatureDetector> f2d = SIFT::create(0, 3, 0.04, 10, 1.6);
	f2d->detect(left, keyPointLeft);
	f2d->detect(right, keyPointRight);

	// ������������
	Mat descriptorLeft, descriptorRight;
	f2d->compute(left, keyPointLeft, descriptorLeft);
	f2d->compute(right, keyPointRight, descriptorRight);

	//ƥ��������
	FlannBasedMatcher matcher;
	vector<DMatch> matchResult;
	matcher.match(descriptorLeft, descriptorRight, matchResult);

	// ����ƥ��ͼ��
	Mat initialMatches;
	drawMatches(left, keyPointLeft, right, keyPointRight, matchResult, initialMatches);
	imshow("���ƥ��", initialMatches);
	waitKey(0);

	// Ѱ�Һõ�ƥ���
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

	// ����ƥ��ͼ�񣨺õ�ƥ��㣩
	Mat goodMatches;
	drawMatches(left, keyPointLeft, right, keyPointRight, goodMatchResult, goodMatches);
	imshow("�õ�ƥ��", goodMatches);
	waitKey(0);

	// ����任����
	vector<Point2f> pointLeft, pointRight;
	for (int i = 0; i < goodMatchResult.size(); i++)
	{
		pointLeft.push_back(keyPointLeft[goodMatchResult[i].queryIdx].pt);
		pointRight.push_back(keyPointRight[goodMatchResult[i].trainIdx].pt);
	}
	Mat transformMat = findHomography(pointRight, pointLeft, RANSAC);

	// Ӱ����׼
	vector<Point2f> rightCornerPoint(4), rightTransformedPoints(4);
	rightCornerPoint[0].x = 0; rightCornerPoint[0].y = 0;
	rightCornerPoint[1].x = 0; rightCornerPoint[1].y = right.rows;
	rightCornerPoint[2].x = right.cols; rightCornerPoint[2].y = right.rows;
	rightCornerPoint[3].x = right.cols; rightCornerPoint[3].y = 0;
	perspectiveTransform(rightCornerPoint, rightTransformedPoints, transformMat);

	// ƴ��
	Mat dst;
	warpPerspective(right, dst, transformMat, Size(700, 700));
	left.copyTo(dst(Rect(0, 0, left.cols, left.rows)));
	line(dst, rightTransformedPoints[0], rightTransformedPoints[1], Scalar(255, 255, 255));
	line(dst, rightTransformedPoints[1], rightTransformedPoints[2], Scalar(255, 255, 255));
	line(dst, rightTransformedPoints[2], rightTransformedPoints[3], Scalar(255, 255, 255));
	line(dst, rightTransformedPoints[3], rightTransformedPoints[0], Scalar(255, 255, 255));
	imshow("ƴ�ӽ��", dst);
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

void shun::RIFT_Matching(string fileL, string fileR)
{
	// step 1: ��תΪ��ɫͼ�����ڻ�������
	Mat opticalColor = imread(fileL, IMREAD_COLOR);
	Mat sarColor = imread(fileR, IMREAD_COLOR);
	// ��ɫͼ��תΪ�Ҷ�ͼ������������ȡ������������
	Mat optical, sar;
	cvtColor(opticalColor, optical, COLOR_BGR2GRAY);
	cvtColor(sarColor, sar, COLOR_BGR2GRAY);

	cout << "read images..." << endl << endl;

	// step 2��ʹ��RIFT��ȡ�������������
	shun::RIFT rift;
	vector<KeyPoint> keyPtsOptical, keyPtsSar;
	Mat desOptical, desSar;
	rift.DetectAndCompute(optical, keyPtsOptical, desOptical);
	rift.DetectAndCompute(sar, keyPtsSar, desSar);

	cout << "RIFT keypoints detected optical: " << keyPtsOptical.size() << endl;
	cout << "RIFT keypoints detected SAR: " << keyPtsSar.size() << endl << endl;

	// step 3����ʼƥ�䣬�ҳ�����ںʹν���
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> initialMatches;
	matcher.knnMatch(desOptical.t(), desSar.t(), initialMatches, 2);

	cout << "initial match: " << initialMatches.size() << " pairs" << endl << endl;

	float maxVal = FLT_MIN, minVal = FLT_MAX;
	for (int i = 0; i < initialMatches.size(); i++)
	{
		const vector<DMatch>& m = initialMatches[i];
		float ratio = m[0].distance / m[1].distance;
		if (ratio > maxVal)
			maxVal = ratio;
		if (ratio < minVal)
			minVal = ratio;
	}

	cout << "distance ratio range: [" << minVal << ", " << maxVal << "]" << endl << endl;

	vector<Point2f> ptsOptical, ptsSar;
	for (int i = 0; i < initialMatches.size(); i++)
	{
		const vector<DMatch>& m = initialMatches[i];
		if (m[0].distance < m[1].distance * 1)
		{
			const Point2f pt1 = keyPtsOptical[m[0].queryIdx].pt;
			const Point2f pt2 = keyPtsSar[m[0].trainIdx].pt;
			ptsOptical.push_back(pt1);
			ptsSar.push_back(pt2);
		}
	}

	// step 4����RANSAC��������任����
	Mat m = findHomography(ptsOptical, ptsSar, RHO);

	cout << "compte Homography matrix using PSOSAC" << endl << endl;

	//// �ӿ���ƻ�������
	//Mat transformMatrix = shun::FastSampleConsensus(smallOptical, smallSar, largeOptical, largeSar, 1000);

	// step 5���޳��ֲ�㣬��λ���С��2����
	vector<Point2f> transformed;
	perspectiveTransform(ptsOptical, transformed, m);

	vector<Point2f> goodMatchesOptical, goodMatchesSar;
	for (int i = 0; i < transformed.size(); i++)
	{
		float x1 = transformed[i].x;
		float y1 = transformed[i].y;
		float x2 = ptsSar[i].x;
		float y2 = ptsSar[i].y;
		float dx = x2 - x1;
		float dy = y2 - y1;

		if (sqrt(dx * dx + dy * dy) < 2.0)
		{
			goodMatchesOptical.push_back(ptsOptical[i]);
			goodMatchesSar.push_back(ptsSar[i]);
		}
	}

	cout << "good matches after outlier removal (position error less than 2 pixels): " << goodMatchesOptical.size() << endl << endl;

	// step 6������ƥ����
	for (int i = 0; i < goodMatchesOptical.size(); i++)
	{
		circle(opticalColor, goodMatchesOptical[i], 2, Scalar(0, 0, 255), -1);
		circle(sarColor, goodMatchesSar[i], 2, Scalar(0, 0, 255), -1);
	}

	imshow("optical", opticalColor);
	imshow("sar", sarColor);

	// step 7: ����ƥ�����
	m = findHomography(goodMatchesOptical, goodMatchesSar, RHO);
	perspectiveTransform(goodMatchesOptical, transformed, m);
	double error = 0;
	for (int i = 0; i < goodMatchesOptical.size(); i++)
	{
		float x1 = transformed[i].x;
		float y1 = transformed[i].y;
		float x2 = goodMatchesSar[i].x;
		float y2 = goodMatchesSar[i].y;
		float dx = x2 - x1;
		float dy = y2 - y1;

		error += dx * dx + dy * dy;
	}
	error = sqrt(error / goodMatchesOptical.size());

	cout << "match error is: " << error << endl;
}

double shun::MutialInfo(cv::Mat patch1, cv::Mat patch2)
{
	assert(patch1.size() == patch2.size());
	assert(patch1.type() == CV_8UC1 && patch1.type() == CV_8UC1);

	Mat hist1 = Mat::zeros(256, 1, CV_32FC1);    // ��Եֱ��ͼ
	Mat hist2 = Mat::zeros(256, 1, CV_32FC1);
	Mat hist12 = Mat::zeros(256, 256, CV_32FC1);    // ����ֱ��ͼ
	
	// ͳ��
	int rows = patch1.rows;
	int cols = patch2.cols;
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int index1 = patch1.ptr<uchar>(r)[c];
			int index2 = patch2.ptr<uchar>(r)[c];

			hist1.ptr<float>(index1)[0]++;
			hist2.ptr<float>(index2)[0]++;
			hist12.ptr<float>(index1)[index2]++;
		}
	}

	// ������
	double e1 = Entropy(hist1);
	double e2 = Entropy(hist2);
	double e12 = Entropy(hist12);

	return e1 + e2 - e12;
}

double shun::Entropy(cv::Mat hist)
{
	double a = sum(hist)[0];
	Mat p = hist / a;    // ����
	Mat log_p;
	log(p + FLT_EPSILON, log_p);    // ע��Ϊ0�����
	Mat temp = p.mul(log_p);

	return -sum(temp)[0];
}

shun::TemplateMatching::TemplateMatching(int patchSize, int searchSize, TEMPLATE_MATCHING_MODE mode)
	: _patchSize(patchSize), _searchSize(searchSize), _mode(mode)
{
}

void shun::TemplateMatching::Execute(std::vector<Point2f>& pts, cv::Mat sensed, cv::Mat reference,
	                                 std::vector<cv::Point2f>& out)
{
	fstream fout("../output/template_matching.txt", ios::out);
	fout << "patch size = " << _patchSize << endl;
	fout << "search size = " << _searchSize << endl;
	fout << "similarity metric = \"" << _modeName[_mode] << "\"" << endl << endl;
	fout << "x_sensed " << "y_sensed " << "x_reference " << "y_reference " << "similarity" << endl;

	// ���쵽0~255��ת��Ϊ8UC1
	Mat temSen, temRef;
	normalize(sensed, temSen, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(reference, temRef, 0, 255, NORM_MINMAX, CV_8UC1);

	int radius = _patchSize / 2;
	int rowsL = temSen.rows;
	int colsL = temSen.cols;
	int rowsR = temRef.cols;
	int colsR = temRef.cols;
	for (int i = 0; i < pts.size(); i++)
	{
		float x = pts[i].x;
		float y = pts[i].y;

		// �Ƿ�Խ��
		int x1 = x - radius;
		int x2 = x + radius;
		int y1 = y - radius;
		int y2 = y + radius;
		if (x1 < 0 || x2 >= colsL || y1 < 0 || y2 >= rowsL)
			continue;

		// sensed patch
		Mat patchL(temSen, Rect(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)));

		// ȷ��������
		x1 = x - _searchSize / 2 - _patchSize / 2;    
		if (x1 < 0) 
			x1 = 0;
		x2 = x + _searchSize / 2 + _patchSize / 2;    
		if (x2 >= colsR) 
			x2 = colsR - 1;
		y1 = y - _searchSize / 2 - _patchSize / 2;
		if (y1 < 0)
			y1 = 0;
		y2 = y + _searchSize / 2 + _patchSize / 2;
		if (y2 >= rowsR)
			y2 = rowsR - 1;
		Mat searchRegion(temRef, Rect(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)));

		// ģ��ƥ��
		double maxSimilarity;
		Point2f p = TemplateMatchingForPoint(patchL, searchRegion, _mode, maxSimilarity);
		p.x += x1 + _patchSize / 2;
		p.y += y1 + _patchSize / 2;

		out.push_back(p);

		fout << x << " " << y << " " << p.x << " " << p.y << " " << maxSimilarity << endl;
	}
	fout.close();
}

shun::WaveletPyramidMathing::WaveletPyramidMathing(int patchSize, int searchSize, int nLayer, string name)
	: _nLayer(nLayer), _searchSize(searchSize), _patchSize(patchSize), _name(name),
	  _pydSensed(nLayer, name), _pydRef(nLayer, name)
{
}

void shun::WaveletPyramidMathing::Execute(std::vector<cv::Point2f>& pts, cv::Mat sensed, cv::Mat reference,
	                                      std::vector<cv::Point2f>& out)
{
	// ��������
	_pydSensed.Build(sensed);
	_pydRef.Build(reference);

	// ������ĵ�ƥ��
	Mat topSensed = _pydSensed.GetCoeffs(_nLayer - 1, "A");
	Mat topRef = _pydRef.GetCoeffs(_nLayer - 1, "A");
	Point2f d = CorePointMatching(topSensed, topRef);
	int dx = d.x, dy = d.y;

	int radius = _patchSize / 2;
	int rowsL = sensed.rows;
	int colsL = sensed.cols;
	for (int i = 0; i < pts.size(); i++)
	{
		float x = pts[i].x;
		float y = pts[i].y;

		// �Ƿ�Խ��
		int x1 = x - radius;
		int x2 = x + radius;
		int y1 = y - radius;
		int y2 = y + radius;
		if (x1 < 0 || x2 >= colsL || y1 < 0 || y2 >= rowsL)
			continue;

		// ��ʼ���ƥ��
		Point2f p(-1, -1);
		for (int l = _nLayer - 1; l >= 0; l--)
		{
			Mat sensedLayer = _pydSensed.GetCoeffs(l, "A");    // ��l��sensed image
			Mat refLayer = _pydRef.GetCoeffs(l, "A");    // ��l��reference image
			int pSize = _patchSize / pow(2, l);    // ��l���ģ���С
			int search = _searchSize / pow(2, l);    // ��l���������Χ�������С 
			int rowsR = refLayer.rows;
			int colsR = refLayer.cols;

			pSize = pSize + ((pSize & 1) ^ 1);    // ��֤����
			search = search + ((search & 1) ^ 1);    // ��֤����

			int x_ = x / pow(2, l);
			int y_ = y / pow(2, l);
			Mat patchSensed(sensedLayer, Rect(x_ - pSize / 2, y_ - pSize / 2, pSize, pSize));

			// ȷ��������
			int x1_ = 0, y1_ = 0, x2_ = 0, y2_ = 0;
			if (l == _nLayer - 1)    // �������ں��ĵ�ƥ����ѵõ���ŵ�ƫ��λ��
			{
				x_ = x_ + d.x;
				y_ = y_ + d.y;
			}
			else    // �����㣬���ϲ�ƥ����Ϊ��ֵ
			{
				x_ = p.x;
				y_ = p.y;
			}

			if (x_ < 0 || x_ >= colsR || y_ < 0 || y_ >= rowsR)    // ���ص���Χ��ĵ�
				break;

			x1 = x_ - search / 2 - pSize / 2;
			if (x1 < 0)
				x1 = 0;
			x2 = x_ + search / 2 + pSize / 2;
			if (x2 >= colsR)
				x2 = colsR - 1;
			y1 = y_ - search / 2 - pSize / 2;
			if (y1 < 0)
				y1 = 0;
			y2 = y_ + search / 2 + pSize / 2;
			if (y2 >= rowsR)
				y2 = rowsR - 1;
			Mat searchRegion(refLayer, Rect(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)));

			if (searchRegion.rows < pSize || searchRegion.cols < pSize)    // ��������û��ģ���
				break;

			double similarity;
			Point2f m = TemplateMatchingForPoint(patchSensed, searchRegion, TM_CCOEFF_NORMED, similarity);
			p.x = m.x + x1 + pSize / 2;
			p.y = m.y + y1 + pSize / 2;

			if (l != 0)
			{
				p.x *= 2;
				p.y *= 2;
			}
		}

		out.push_back(p);
	}
}

cv::Point2f shun::WaveletPyramidMathing::CorePointMatching(cv::Mat sensed, cv::Mat reference)
{
	int x = sensed.cols / 2;
	int y = sensed.rows / 2;
	int pSize = _patchSize / pow(2, _nLayer - 1);
	pSize = pSize + ((pSize & 1) ^ 1);    // ��֤����

	Mat patchSensed(sensed, Rect(x - pSize / 2, y - pSize / 2, pSize, pSize));

	double val;
	Point2f p = TemplateMatchingForPoint(patchSensed, reference, TM_CCOEFF_NORMED, val);

	p.x += pSize / 2;
	p.y += pSize / 2;

	p.x -= x;
	p.y -= y;

	return p;
}
