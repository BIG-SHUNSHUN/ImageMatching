// ImageMatching.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include "utils.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "wavelet.h"
#include "matchfuncs.h"


using namespace cv;
using namespace std;

#include "phase.h"
#include "features.h"
#include "gamma.h"

void PhaseCongruencyDemo()
{
	string inputFileName = "../image/fig.png";
	Mat image = imread(inputFileName, IMREAD_GRAYSCALE);
	if (image.empty())
	{
		cout << "Cannot read image file " << inputFileName << endl;
		return;
	}

	int start = getTickCount();
	PhaseCongruency pc;
	pc.Calc(image);

	Mat edge, corner;
	pc.Feature(edge, corner);
	int end = getTickCount();
	cout << (end - start) / getTickFrequency() << endl;

	Mat orient;
	pc.Orientation(orient);

	// namedWindow("image");
	imshow("edges", edge);
	imshow("corners", corner);
	waitKey(0);
	destroyAllWindows();
}

void RIFTDemo()
{
	string sarFile = "../image/pair3.tif";
	string opticalFile = "../image/pair4.tif";

	Mat sar = imread(sarFile, IMREAD_GRAYSCALE);
	Mat optical = imread(opticalFile, IMREAD_GRAYSCALE);

	shun::RIFT rift;
	vector<KeyPoint> keyPtsSar, keyPtsOptical;
	Mat desSar, desOptical;

	rift.DetectAndCompute(sar, keyPtsSar, desSar);
	rift.DetectAndCompute(optical, keyPtsOptical, desOptical);
}

void GAMMADemo()
{
	/*string strParams = "D:\\working_zone\\data\\GAMMA_SAR\\MLI\\20201130.mli.par";
	string strBinary = "D:\\working_zone\\data\\GAMMA_SAR\\MLI\\20201130.mli";*/

	string strParams = "E:\\data\\SAR\\DESC\\rmli.par";
	string strBinary = "E:\\data\\SAR\\DESC\\ave.rmli";

	//string strParams = "E:\\data\\song-data\\S1a\\20201120.mli.par";
	//string strBinary = "E:\\data\\song-data\\S1a\\20201120.mli";
	
	shun::GammaImage image;
	image.Read(strParams, strBinary);
	image.Show();

	
	
	//string pointStr = "E:\\data\\SAR\\ASC\\point";
	//vector<Point> ps_points = shun::ReadPersistentScatterPoint(pointStr);

	//shun::DrawPersistentPoints(image.GetMat(), ps_points);

	//imwrite("D:/sar.jpg", image);
}

void NonlinearDiffsionDemo()
{
	string path = "C:\\Users\\shunshun\\Desktop\\programming\\need_to_study\\HAPCG-Multimodal-matching-main\\Images\\1-1光照差异.png";
	Mat img = imread(path, IMREAD_GRAYSCALE);

	shun::NonlinearSpace space(3, 2.0);
	space.Generate(img);

	for (int i = 0; i < 3; i++)
	{
		Mat layer = space.GetLayer(i);
		imshow("Layer", layer);
		waitKey(0);
	}
	destroyAllWindows();

	PhaseCongruency pc;
	pc.Calc(space.GetLayer(0));
	imshow("pc", pc.pcSum());
	waitKey(0);
	destroyAllWindows();
}

void RIFTMatchDemo()
{
	// step 1：光学影像和SAR影像
	string opticalFile = "../image/pair4.tif";
	string sarFile = "../image/pair3.tif";
	// 先转为彩色图，便于画特征点
	Mat opticalColor = imread(opticalFile, IMREAD_COLOR);
	Mat sarColor = imread(sarFile, IMREAD_COLOR);
	// 彩色图像转为灰度图，用于特征提取及描述符构建
	Mat optical, sar;
	cvtColor(opticalColor, optical, COLOR_BGR2GRAY);
	cvtColor(sarColor, sar, COLOR_BGR2GRAY);

	cout << "read images..." << endl << endl;

	// step 2：使用RIFT提取特征点和描述符
	shun::RIFT rift;
	vector<KeyPoint> keyPtsOptical, keyPtsSar;
	Mat desOptical, desSar;
	rift.DetectAndCompute(optical, keyPtsOptical, desOptical);
	rift.DetectAndCompute(sar, keyPtsSar, desSar);
	
	cout << "RIFT keypoints detected optical: " << keyPtsOptical.size() << endl;
	cout << "RIFT keypoints detected SAR: " << keyPtsSar.size() << endl << endl;

	// step 3：初始匹配，找出最近邻和次近邻
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

	// step 4：用RANSAC方法计算变换矩阵
	Mat m = findHomography(ptsOptical, ptsSar, RHO);

	cout << "compte Homography matrix using PSOSAC" << endl << endl;

	//// 接口设计还需完善
	//Mat transformMatrix = shun::FastSampleConsensus(smallOptical, smallSar, largeOptical, largeSar, 1000);
	
	// step 5：剔除粗差点，点位误差小于2像素
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

	// step 6：画出匹配结果
	for (int i = 0; i < goodMatchesOptical.size(); i++)
	{
		circle(opticalColor, goodMatchesOptical[i], 2, Scalar(0, 0, 255), -1);
		circle(sarColor, goodMatchesSar[i], 2, Scalar(0, 0, 255), -1);
	}

	imshow("optical", opticalColor);
	imshow("sar", sarColor);

	// step 7: 计算匹配误差
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

void HAPCG_Demo()
{
	// step 1：光学影像和SAR影像
	string opticalFile = "../image/pair4.tif";
	string sarFile = "../image/pair3.tif";
	// 先转为彩色图，便于画特征点
	Mat opticalColor = imread(opticalFile, IMREAD_COLOR);
	Mat sarColor = imread(sarFile, IMREAD_COLOR);
	// 彩色图像转为灰度图，用于特征提取及描述符构建
	Mat optical, sar;
	cvtColor(opticalColor, optical, COLOR_BGR2GRAY);
	cvtColor(sarColor, sar, COLOR_BGR2GRAY);

	cout << "read images..." << endl << endl;

	shun::NonlinearSpace spaceOptical, spaceSar;
	spaceOptical.Generate(optical);
	spaceSar.Generate(sar);


}

int main(int argc, char** argv)
{
	PhaseCongruencyDemo();
	RIFTMatchDemo();

	string fileL = "../image/pair4.tif";
	string fileR = "../image/52_073_rgbx.tif";

	Mat imgL = imread(fileL, IMREAD_COLOR);
	Mat imgR = imread(fileR, IMREAD_COLOR);

	Mat imgL_gray, imgR_gray;
	cvtColor(imgL, imgL_gray, COLOR_BGR2GRAY);
	cvtColor(imgR, imgR_gray, COLOR_BGR2GRAY);

	{
		shun::HarrisDetector detector;
		vector<Point2f> pts;
		detector.DetectAndCompute(imgL_gray, pts, 1);

		Mat imgDraw;
		imgL.copyTo(imgDraw);
		shun::DrawFeaturePoints(imgDraw, pts);
		imshow("harris", imgDraw);
		waitKey(0);
	}

	{
		shun::ShiTomashiDetector detector(500, true);
		vector<Point2f> pts;
		detector.DetectAndCompute(imgL_gray, pts);

		Mat imgDraw;
		imgL.copyTo(imgDraw);
		shun::DrawFeaturePoints(imgDraw, pts);
		imshow("ShiTomashi", imgDraw);
		waitKey(0);
	}

	destroyAllWindows();

	//Mat imgL_Double, imgR_Double;
	//imgL.convertTo(imgL_Double, CV_64FC1);
	//imgR.convertTo(imgR_Double, CV_64FC1);

	//shun::WaveletPyramid pyramid(imgL_gray, 6, "haar");
	//pyramid.ShowImage(0, nullptr);
	//pyramid.ShowImage(1, "A");
	//pyramid.ShowImage(2, "A");
	//pyramid.ShowImage(3, "A");
	//pyramid.ShowImage(4, "A");
	//pyramid.ShowImage(5, "A");
	// pyramid.ShowImage(3, "D");

	//// 模板匹配
	//Point pts(imgL.cols / 2 + 100, imgR.rows / 2);
	//Point result = TemplateMatchingForPoint(imgL_gray, imgR_gray, pts, 51, TM_CCORR_NORMED);

	//circle(imgL, pts, 5, Scalar(0, 0, 255), -1);
	//circle(imgR, result, 5, Scalar(0, 0, 255), -1);

	//imshow("imgL", imgL);
	//imshow("imgR", imgR);
	//waitKey(0);
	//destroyAllWindows();

	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
