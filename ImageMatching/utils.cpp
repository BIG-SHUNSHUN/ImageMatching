#include "utils.h"
#include <fstream>

void shun::Trim(string & s, char ch)
{
	int low = s.find_first_not_of(ch);
	if (low != 0)
	{
		s.erase(0, low);
	}
	int high = s.find_last_not_of(ch);
	if (high != s.size() - 1)
	{
		s.erase(high + 1);
	}
}

void shun::Split(vector<string>& splitString, string&s, char ch)
{
	//把string变成stringstream
	stringstream sstream(s);
	string temp;
	while (getline(sstream, temp, ch))
	{
		if (temp.size() == 0) continue;
		Trim(temp);    //删除首尾的空格
		Trim(temp, '}');    //删除首尾的 }
		splitString.push_back(temp);
	}
}

void shun::CircShift(Mat & inAndOut, Size step)
{
	int stepX = step.width;
	int stepY = step.height;
	int rows = inAndOut.rows;
	int cols = inAndOut.cols;
	stepX = stepX % cols;
	stepY = stepY % rows;

	if (cols == 0 || rows == 0)
		return;

	Mat tmp;
	tmp.create(inAndOut.size(), inAndOut.type());

	Mat s1(inAndOut, Rect(0, 0, cols - stepX, rows - stepY));
	Mat s2(inAndOut, Rect(cols - stepX, 0, stepX, rows - stepY));
	Mat s3(inAndOut, Rect(cols - stepX, rows - stepY, stepX, stepY));
	Mat s4(inAndOut, Rect(0, rows - stepY, cols - stepX, stepY));

	Mat d1(tmp, Rect(0, 0, stepX, stepY));
	Mat d2(tmp, Rect(stepX, 0, cols - stepX, stepY));
	Mat d3(tmp, Rect(stepX, stepY, cols - stepX, rows - stepY));
	Mat d4(tmp, Rect(0, stepY, stepX, rows - stepY));

	s1.copyTo(d3);
	s2.copyTo(d4);
	s3.copyTo(d1);
	s4.copyTo(d2);

	inAndOut = tmp;
}

void shun::FFTShift(Mat & inAndOut)
{
	int rows = inAndOut.rows;
	int cols = inAndOut.cols;

	CircShift(inAndOut, Size(cols / 2, rows / 2));
}

void shun::IFFTShift(Mat & inAndOut)
{
	int rows = inAndOut.rows;
	int cols = inAndOut.cols;

	CircShift(inAndOut, Size((cols + 1) / 2, (rows + 2) / 2));
}

void shun::MeshGrid(Mat x, Mat y, Mat & matX, Mat & matY)
{
	int ny = y.rows;
	int nx = x.rows;
	cv::repeat(x, ny, 1, matX);
	cv::repeat(y.t(), 1, nx, matY);
}

void shun::CountTime(void(*func)())
{
	int start = getTickCount();
	func();
	int end = getTickCount();
	cout << (end - start) / getTickFrequency() << endl;
}

double shun::ValuePercent(Mat in, double prec)
{
	// OpenCV计算直方图函数不支持CV_64F，要转为CV_32F
	Mat src ;
	if (in.type() == CV_64FC1)
		in.convertTo(src, CV_32FC1);
	else
		src = in;

	// 计算最大值和最小值
	double minVal = 0, maxVal = 0;
	minMaxLoc(in, &minVal, &maxVal);

	// 计算直方图
	int rows = in.rows;
	int cols = in.cols;
	float ranges[] = { minVal, maxVal + 0.0001 };    // 数据范围
	const float* histRange = { ranges };
	int nbins = rows * cols / 2;    // 直方图柱子个数
	Mat hist;
	calcHist(&src, 1, 0, Mat(), hist, 1, &nbins, &histRange);

	// 累积计数
	int count = 0;
	double thresh = 0;
	for (int i = 0; i < nbins; i++)
	{
		int n = hist.at<float>(i, 0);
		count += static_cast<int>(n);    

		// 数量达到要求
		if (count >= rows * cols * prec)
		{
			thresh = (maxVal - minVal) / nbins * (i + 1) + minVal;
			break;
		}
	}

	return thresh;
}

void shun::Atan2ForMat(cv::Mat inX, cv::Mat inY, cv::Mat & out)
{
	assert(inX.size() == inY.size());

	Mat dx, dy;
	if (inX.type() != CV_64FC1)
		inX.convertTo(dx, CV_64FC1);
	else
		dx = inX;
	if (inY.type() != CV_64FC1)
		inY.convertTo(dy, CV_64FC1);
	else
		dy = inY;

	int rows = inX.rows;
	int cols = inY.cols;
	out.create(inX.size(), CV_64FC1);
	for (int r = 0; r < rows; r++)
	{
		double* ptrX = dx.ptr<double>(r);
		double* ptrY = dy.ptr<double>(r);
		double* outPtr = out.ptr<double>(r);
		for (int c = 0; c < cols; c++)
		{
			outPtr[c] = atan2(ptrY[c], ptrX[c]);
		}
	}
}
