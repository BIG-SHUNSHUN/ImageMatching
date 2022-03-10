#include "utils.h"
#include <fstream>

void Trim(string & s, char ch)
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

void Split(vector<string>& splitString, string&s, char ch)
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

void CircShift(Mat & inAndOut, Size step)
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

void FFTShift(Mat & inAndOut)
{
	int rows = inAndOut.rows;
	int cols = inAndOut.cols;

	CircShift(inAndOut, Size(cols / 2, rows / 2));
}

void IFFTShift(Mat & inAndOut)
{
	int rows = inAndOut.rows;
	int cols = inAndOut.cols;

	CircShift(inAndOut, Size((cols + 1) / 2, (rows + 2) / 2));
}

void MeshGrid(Mat x, Mat y, Mat & matX, Mat & matY)
{
	int ny = y.rows;
	int nx = x.rows;
	cv::repeat(x, ny, 1, matX);
	cv::repeat(y.t(), 1, nx, matY);
}

void CountTime(void(*func)())
{
	int start = getTickCount();
	func();
	int end = getTickCount();
	cout << (end - start) / getTickFrequency() << endl;
}
