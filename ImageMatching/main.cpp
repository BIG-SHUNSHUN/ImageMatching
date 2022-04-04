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
using namespace shun;

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
	pc.SetParams(4, 6);
	pc.Prepare(image);

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
	pc.SetParams(4, 6);
	pc.Prepare(space.GetLayer(0));
	imshow("pc", pc.pcSum());
	waitKey(0);
	destroyAllWindows();
}

void RIFTMatchDemo()
{
	// 光学影像和SAR影像
	string opticalFile = "../image/pair4.tif";
	string sarFile = "../image/pair3.tif";

	RIFT_Matching(opticalFile, sarFile);
}

void HAPCG_Demo()
{
	
}

void OrienCrossDetectDemo()
{
	string file = "../image/pair4.tif";
	Mat img = imread(file, IMREAD_GRAYSCALE);
	WaveletPyramid pyramid(img, 6, "bior1.1");

	Mat h = pyramid.GetCoeffs(1, "H");
	Mat v = pyramid.GetCoeffs(1, "V");

	// Canny 边缘提取
	Mat A;
	normalize(pyramid.GetCoeffs(1, "A"), A, 0, 255, NORM_MINMAX, CV_8UC1);
	Mat edge;
	Canny(A, edge, 80, 120);
	// sobel梯度
	Mat dx, dy;
	Sobel(A, dx, CV_32FC1, 0, 1);
	Sobel(A, dy, CV_32FC1, 1, 0);
	Mat grad_sobel;
	magnitude(dx, dy, grad_sobel);
	normalize(grad_sobel, grad_sobel, 0, 1, NORM_MINMAX);

	// 小波梯度
	Mat g_map;
	magnitude(h, v, g_map);
	Mat grad_wavelet;
	normalize(g_map, grad_wavelet, 0, 255, NORM_MINMAX, CV_8UC1);
	equalizeHist(grad_wavelet, grad_wavelet);

	Mat d_map;
	Atan2ForMat(h, v, d_map);

	double thresh = ValuePercent(g_map, 0.4);

	Mat mask = g_map > thresh;

	Mat edge1 = OrientCrossDetection(g_map, d_map, mask);
}

void TemplatMatchingDemo()
{
	string fileL = "../image/cityL.tif";
	string fileR = "../image/cityR.tif";

	Mat imgL = imread(fileL, IMREAD_COLOR);
	Mat imgR = imread(fileR, IMREAD_COLOR);

	Mat imgL_gray, imgR_gray;
	cvtColor(imgL, imgL_gray, COLOR_BGR2GRAY);
	cvtColor(imgR, imgR_gray, COLOR_BGR2GRAY);

	Mat edgeL, edgeR;
	Canny(imgL_gray, edgeL, 80, 150);
	Canny(imgR_gray, edgeR, 80, 150);

	ShiTomashiDetector detector(500, true);
	vector<Point2f> pts;
	detector.DetectAndCompute(imgL_gray, pts, 51);

	//vector<Point2f> matchPoints;
	//TemplateMatching m(51, 21);
	//m.Execute(pts, edgeL, edgeR, matchPoints);

	vector<Point2f> matchPoints;
	WaveletPyramidMathing m(51, 31);
	m.Execute(pts, edgeL, edgeR, matchPoints);

	DrawFeaturePoints(imgL, pts);
	DrawFeaturePoints(imgR, matchPoints);

	imshow("imgL", imgL);
	imshow("imgR", imgR);
	waitKey(0);

	destroyAllWindows();
}

int main(int argc, char** argv)
{
	PhaseCongruencyDemo();
	TemplatMatchingDemo();
	// RIFTMatchDemo();

	string fileL = "../image/cityL.tif";
	string fileR = "../image/cityR.tif";

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
	//pyramid.Show(0, nullptr);
	//pyramid.Show(1, "A");
	//pyramid.Show(2, "A");
	//pyramid.Show(3, "A");
	//pyramid.Show(4, "A");
	//pyramid.Show(5, "A");
	//pyramid.Show(3, "D");

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
