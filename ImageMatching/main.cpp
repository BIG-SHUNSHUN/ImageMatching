// ImageMatching.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include "utils.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "wavelet.h"


using namespace cv;
using namespace std;

#include "phase.h"
#include "features.h"

int PhaseCongruencyDemo()
{
	string inputFileName = "../image/fig.png";
	Mat image = imread(inputFileName, IMREAD_GRAYSCALE);
	if (image.empty())
	{
		cout << "Cannot read image file " << inputFileName << endl;
		return -1;
	}

	//auto size = image.size() * 3;
	//resize(image, image, size);
	int64 start = getTickCount();

	PhaseCongruency pc;
	pc.Calc(image);

	Mat edge, corner;
	pc.Feature(edge, corner);

	int64 end = getTickCount();
	cout << "time = " << (end - start) / getTickFrequency() * 1000 << endl;

	// namedWindow("image");
	imshow("edges", edge);
	imshow("corners", corner);
	waitKey(0);
	destroyAllWindows();

	return 0;
}

int main(int argc, char** argv)
{
	PhaseCongruencyDemo();


	string fileL = "../image/pair4.tif";
	string fileR = "../image/52_073_rgbx.tif";

	Mat imgL = imread(fileL, IMREAD_COLOR);
	Mat imgR = imread(fileR, IMREAD_COLOR);

	Mat imgL_gray, imgR_gray;
	cvtColor(imgL, imgL_gray, COLOR_BGR2GRAY);
	cvtColor(imgR, imgR_gray, COLOR_BGR2GRAY);

	
	{
		shun::HarrisDetector detector;
		vector<Point> pts;
		detector.DetectAndCompute(imgL_gray, pts);

		Mat imgDraw;
		imgL.copyTo(imgDraw);
		shun::DrawFeaturePoints(imgDraw, pts);
		imshow("harris", imgDraw);
		waitKey(0);
	}

	{
		shun::ShiTomashiDetector detector(500, true);
		vector<Point> pts;
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


int gamma()
{
	//// 
	string str = "E:\\data\\SAR\\ASC\\utm.par";
	GammaImagePara par = ReadGammaImageUTMPara(str);
	Mat image = ReadGammaImage("E:\\data\\SAR\\ASC\\utm.rmli", par);

	DisplayImage(image);

	convertScaleAbs(image, image, 255);

	string pointStr = "D:\\data\\SAR\\ASC\\point";
	vector<Point> ps_points = ReadPersistentScatterPoint(pointStr);

	DrawPersistentPoints(image, ps_points);

	imwrite("D:/sar.jpg", image);

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
