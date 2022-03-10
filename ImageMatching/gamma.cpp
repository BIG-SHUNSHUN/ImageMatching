#include "gamma.h"
#include "utils.h"
#include <fstream>

using namespace std;
using namespace cv;

shun::GAMMA_FORMAT shun::GammaImage::Format(string str)
{
	Trim(str);
	if (str == "INT")
		return GAMMA_FORMAT::INT;
	else if (str == "FLOAT")
		return GAMMA_FORMAT::FLOAT;
	else if (str == "DOUBLE")
		return GAMMA_FORMAT::DOUBLE;
}


shun::GammaImage::GammaImage()
{
}

shun::GammaImage::~GammaImage()
{
}

void shun::GammaImage::Read(const string & dirParams, const string & dirData)
{
	ReadParams(dirParams);
	ReadBinary(dirData);
}

void shun::GammaImage::Show()
{
	if (_imgToShow.empty())
	{
		double minVal, maxVal;
		minMaxLoc(_img, &minVal, &maxVal);

		// 计算直方图 
		int rows = _img.rows;
		int cols = _img.cols;
		float ranges[] = { minVal, maxVal + 0.0000001 };    // 像素数值范围
		const float* histRange = { ranges };
		int nbins = rows * cols / 2;    // 柱子个数
		Mat hist;
		calcHist(&_img, 1, 0, Mat(), hist, 1, &nbins, &histRange);

		// 计算linear 2%拉伸的上界值和下界值
		int lowerBound = 0;
		int upperBound = 0;
		int n = rows * cols * 0.02;
		int sum = 0;
		for (int r = 0; r < nbins; r++)
		{
			float* ptr = hist.ptr<float>(r);
			sum += static_cast<int>(ptr[0]);
			if (sum >= n)
			{
				lowerBound = (r + 1) * (maxVal - minVal) / nbins + minVal;
				break;
			}
		}
		sum = 0;
		for (int r = nbins - 1; r >= 0; r--)
		{
			float* ptr = hist.ptr<float>(r);
			sum += static_cast<int>(ptr[0]);
			if (sum >= n)
			{
				upperBound = r * (maxVal - minVal) / nbins + minVal;
				break;
			}
		}

		// 拉伸
		_imgToShow.create(_img.size(), CV_8UC1);
		for (int r = 0; r < rows; r++)
		{
			float* ptrImg = _img.ptr<float>(r);
			uchar* ptrShow = _imgToShow.ptr<uchar>(r);
			for (int c = 0; c < cols; c++)
			{
				float val = ptrImg[c];
				if (val <= lowerBound)    // 小于最小边界设为0
				{
					ptrShow[c] = 0;
				}
				else if (val >= upperBound)    // 大于最小边界设为255
				{
					ptrShow[c] = 255;
				}
				else    // 线性拉伸
				{
					ptrShow[c] = static_cast<uchar>((val - lowerBound) / (upperBound - lowerBound) * 255);
				}
			}
		}
	}

	imshow("GAMMA Image", _imgToShow);
	waitKey(0);
	destroyWindow("GAMMA Image");
}

void shun::GammaImage::ReadParams(const string & fileName)
{
	ifstream fin(fileName, ios::in);
	string line;
	while (getline(fin, line))
	{
		int index = line.find_first_of(':');
		if (index >= 0)
		{
			string sub = line.substr(0, index);
			if (sub == "width" || sub == "range_samples")
			{
				_params.samples = stoi(line.substr(index + 1));
			}
			else if (sub == "nlines" || sub == "azimuth_lines")
			{
				_params.lines = stoi(line.substr(index + 1));
			}
			else if (sub == "image_format")
			{
				_params.format = Format(line.substr(index + 1));
			}
		}
	}
	fin.close();
}

void shun::GammaImage::ReadBinary(const string & fileName)
{
	ifstream fin(fileName, ios::binary);

	char buffer[4];
	int rows = _params.lines;
	int cols = _params.samples;
	Mat image(rows, cols, CV_32F);
	for (int r = 0; r < rows; r++)
	{
		float* ptr = image.ptr<float>(r);
		for (int c = 0; c < cols; c++)
		{
			fin.read(buffer, 4);
			float pixelVal = reversebytes_uint32t(*(uint32_t*)buffer);
			ptr[c] = pixelVal;
		}
	}

	fin.close();
	_img = image;
}

vector<Point> shun::ReadPersistentScatterPoint(const string & str)
{
	ifstream fin(str, ios::binary | ios::in);
	//char buffer[4];
	char buffer[8];
	vector<Point> points;
	while (fin.read(buffer, 8))
	{
		//short x = *(short*)buffer;
		//short y = *(short*)(buffer + 2);
		int x = *(int*)buffer;
		int y = *(int*)(buffer + 4);

		x = reversebytes_uint32t(*(uint32_t*)&x);
		y = reversebytes_uint32t(*(uint32_t*)&y);
		//x = reversebytes_uint16(*(uint16_t*)&x);
		//y = reversebytes_uint16(*(uint16_t*)&y);

		Point p(x / 1000, y / 1000);
		points.push_back(p);
	}
	fin.close();
	return points;
}

void shun::DrawPersistentPoints(Mat image, const vector<Point>& points)
{
	Mat disp;
	image.copyTo(disp);
	cvtColor(disp, disp, COLOR_GRAY2BGR);

	for (int i = 0; i < points.size(); i++)
	{
		int x = points[i].x;
		int y = points[i].y;

		circle(disp, points[i], 2, Scalar(0, 0, 255), -1);
	}

	imshow("PS", disp);
	waitKey(0);
	destroyWindow("PS");
}