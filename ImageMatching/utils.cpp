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

GAMMA_FORMAT Format(string str)
{
	Trim(str);
	if (str == "INT")
		return GAMMA_FORMAT::INT;
	else if (str == "FLOAT")
		return GAMMA_FORMAT::FLOAT;
	else if (str == "DOUBLE")
		return GAMMA_FORMAT::DOUBLE;
}

GammaImagePara ReadGammaImagePara(const string& fileName)
{
	GammaImagePara par;
	ifstream fin(fileName, ios::in);
	string line;
	while (getline(fin, line))
	{
		int index = line.find_first_of(':');
		if (index >= 0)
		{
			string sub = line.substr(0, index);
			if (sub == "range_samples")
			{
				par.samples = stoi(line.substr(index + 1));
			}
			else if (sub == "azimuth_lines")
			{
				par.lines = stoi(line.substr(index + 1));
			}
			else if (sub == "image_format")
			{
				par.format = Format(line.substr(index + 1));
			}
		}
	}
	fin.close();
	return par;
}

GammaImagePara ReadGammaImageUTMPara(const string & fileName)
{
	GammaImagePara par;
	ifstream fin(fileName, ios::in);
	string line;
	while (getline(fin, line))
	{
		int index = line.find_first_of(':');
		if (index >= 0)
		{
			string sub = line.substr(0, index);
			if (sub == "width")
			{
				par.samples = stoi(line.substr(index + 1));
			}
			else if (sub == "nlines")
			{
				par.lines = stoi(line.substr(index + 1));
			}
		}
	}
	par.format = GAMMA_FORMAT::FLOAT;
	fin.close();
	return par;
}

Mat ReadGammaImage(const string& fileName, GammaImagePara& par)
{
	ifstream fin(fileName, ios::binary);
	char buffer[4];
	Mat image(par.lines, par.samples, CV_32F);
	for (int row = 0; row < par.lines; row++)
	{
		for (int col = 0; col < par.samples; col++)
		{
			fin.read(buffer, 4);
			float pixelVal = reversebytes_uint32t(*(uint32_t*)buffer);
			image.at<float>(row, col) = pixelVal;
		}
	}
	fin.close();
	return image;
}

void DisplayImage(Mat image)
{
	float minVal = 0, maxVal = 0;
	bool set = false;
	int rows = image.rows, cols = image.cols;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			float pixelVal = image.at<float>(row, col);
			if (pixelVal != 0 && !set)
			{
				minVal = pixelVal;
				maxVal = pixelVal;
				set = true;
			}

			if (pixelVal == 0)
				continue;
			if (pixelVal < minVal)
				minVal = pixelVal;
			if (pixelVal > maxVal)
				maxVal = pixelVal;
		}
	}

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			float pixelVal = (image.at<float>(row, col) - minVal) / (maxVal - minVal);
			image.at<float>(row, col) = pixelVal;
		}
	}

	imshow("image", image);
	waitKey(0);
	destroyWindow("image");
}

vector<Point> ReadPersistentScatterPoint(const string & str)
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

void DrawPersistentPoints(Mat image, const vector<Point>& points)
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
