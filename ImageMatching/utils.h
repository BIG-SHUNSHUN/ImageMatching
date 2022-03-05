#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

enum GAMMA_FORMAT
{
	INT,
	FLOAT,
	DOUBLE
};

struct GammaImagePara
{
	int lines;
	int samples;
	GAMMA_FORMAT format;
};

void Trim(string & s, char ch = ' ');

void Split(vector<string>& splitString, string&s, char ch);

GAMMA_FORMAT Format(string str);

GammaImagePara ReadGammaImagePara(const string& fileName);

GammaImagePara ReadGammaImageUTMPara(const string& fileName);

inline uint32_t reversebytes_uint32t(uint32_t value) {
	return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
		(value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
}

inline uint16_t reversebytes_uint16(uint16_t value)
{
	return (value & 0x00FFU) << 8 | (value & 0xFF00U) >> 8;
}

Mat ReadGammaImage(const string& fileName, GammaImagePara& par);

void DisplayImage(Mat image);

vector<Point> ReadPersistentScatterPoint(const string& str);

void DrawPersistentPoints(Mat image, const vector<Point>& points);

void CircShift(Mat& inAndOut, Size step);

void FFTShift(Mat& inAndOut);

void IFFTShift(Mat& inAndOut);

void MeshGrid(Mat x, Mat y, Mat& matX, Mat& matY);