#include "phase.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <corecrt_math_defines.h>
#include "utils.h"

using namespace cv;
using namespace std;

void Helper(int n, vector<double>& arr)
{
	if (n % 2 == 1)
	{
		int gap = (n - 1) / 2;
		for (int i = 0; i < arr.size(); i++)
		{
			arr[i] = (i - gap) / static_cast<double>((n - 1));
		}
	}
	else
	{
		int gap = n / 2;
		for (int i = 0; i < arr.size(); i++)
		{
			arr[i] = (i - gap) / static_cast<double>(n);
		}
	}
}

Mat LowPassFilter(Size size, double cutOff, int n)
{
	int rows = size.height;
	int cols = size.width;

	vector<double> xRange(cols, 0);
	vector<double> yRange(rows, 0);
	
	Helper(cols, xRange);
	Helper(rows, yRange);

	Mat radius = Mat::zeros(rows, cols, ELEM_TYPE);
	for (int r = 0; r < rows; r++)
	{
		double* ptr = radius.ptr<double>(r);
		for (int c = 0; c < cols; c++)
		{
			double x = xRange[c];
			double y = yRange[r];
			ptr[c] = sqrt(x * x + y * y);
		}
	}

	pow(radius / cutOff, 2 * n, radius);
	radius = 1.0 / (radius + 1.0);
	IFFTShift(radius);
	
	return radius;
}

// Making a _filter
// src & dst arrays of equal size & type
PhaseCongruency::PhaseCongruency(int _nScale, int _nOrient)
	: _nScale(_nScale), _nOrient(_nOrient),
	_eo(_nScale, vector<Mat>(_nOrient)), //s*o
	_filter(_nScale * _nOrient),
	_pc(_nOrient)
{
}

//Phase congruency calculation
void PhaseCongruency::Calc(Mat src)
{
	Initialize(src.size());

	// expand input image to optimal size
	Mat planes[] = { Mat_<double>(src), Mat::zeros(src.size(), ELEM_TYPE) };
	Mat imageFFT;
	merge(planes, 2, imageFFT);         // Add to the expanded another plane with zeros
	dft(imageFFT, imageFFT);            // this way the result may fit in the source matrix

	Mat pcSum = Mat::zeros(src.size(), ELEM_TYPE);
	Mat complex[2];
	Mat An;
	Mat sorted;
	Mat maxAn = Mat::zeros(src.size(), ELEM_TYPE);
    for (int o = 0; o < _nOrient; o++)
    {
		Mat sumE = Mat::zeros(src.size(), ELEM_TYPE);
		Mat sumO = Mat::zeros(src.size(), ELEM_TYPE);
		Mat sumAn = Mat::zeros(src.size(), ELEM_TYPE);
		Mat energy = Mat::zeros(src.size(), ELEM_TYPE);
		double tau = 0;
		
        for (int s = 0; s < _nScale; s++)
        {
            mulSpectrums(imageFFT, _filter[o * _nScale + s], _eo[s][o], 0); // Convolution
			idft(_eo[s][o], _eo[s][o], DFT_SCALE);

			split(_eo[s][o], complex);
            magnitude(complex[0], complex[1], An);

			sumAn += An;
			sumE += complex[0];
			sumO += complex[1];

			if (s == 0)
			{
				if (_noiseMethod == -1)
				{
					Mat reShape = sumAn.reshape(1, 1);
					cv::sort(reShape, sorted, SortFlags::SORT_ASCENDING);
					if (sorted.cols % 2 == 0)
					{
						double v1 = sorted.at<double>(0, sorted.cols / 2);
						double v2 = sorted.at<double>(0, sorted.cols / 2 - 1);
						tau = (v1 + v2) * 0.5;
					}
					else
					{
						tau = sorted.at<double>(0, sorted.cols / 2);
					}
					tau = tau / sqrt(log(4));
				}
				else
				{
				}
				An.copyTo(maxAn);
			}
			else
			{
				max(maxAn, An, maxAn);
			}
        } // next scale

 		Mat xEnergy, meanE, meanO;
		magnitude(sumE, sumO, xEnergy); 
		xEnergy += _epsilon;
		divide(sumE, xEnergy, meanE);
		divide(sumO, xEnergy, meanO);

		for (int s = 0; s < _nScale; s++)
		{
			split(_eo[s][o], complex);
			energy += complex[0].mul(meanE) + complex[1].mul(meanO) - 
				abs(complex[0].mul(meanO) - complex[1].mul(meanE));
		}

        double T = 0;
		if (_noiseMethod >= 0)
		{
			T = _noiseMethod;
		}
		else
		{
			double totalTau = tau * (1 - pow(1 / _mult, _nScale)) / (1 - (1 / _mult));
			double noiseMean = totalTau * sqrt(M_PI / 2);
			double noiseSigma = totalTau * sqrt((4 - M_PI) / 2);
			T = noiseMean + _k * noiseSigma;
		}

        max(energy -= T, 0.0, energy);

		Mat width = (sumAn / (maxAn +_epsilon) - 1.0) / (_nScale - 1.0);
		//cv::divide(sumAn, maxAn + _epsilon, width);
		//width = (width - 1) / (_nScale - 1.0);

		Mat weight;
		exp((_cutOff - width) * _g, weight);
		weight = 1.0 / (1.0 + weight);

		cv::divide(weight.mul(energy), sumAn, _pc[o]);

		pcSum += _pc[o];
    }//orientation
	_pcSum = pcSum;
}

void PhaseCongruency::Feature(cv::Mat& outEdges, cv::Mat& outCorners)
{
	Mat covx2 = Mat::zeros(_size, ELEM_TYPE);
	Mat covy2 = Mat::zeros(_size, ELEM_TYPE);
	Mat covxy = Mat::zeros(_size, ELEM_TYPE);
	for (int o = 0; o < _nOrient; o++)
	{
		double angle = o * M_PI / _nOrient;
		Mat tmpX = _pc[o] * cos(angle);
		Mat tmpY = _pc[o] * sin(angle);

		covx2 += tmpX.mul(tmpX);
		covy2 += tmpY.mul(tmpY);
		covxy += tmpX.mul(tmpY);
	}

	covx2 = covx2 / _nOrient * 2;
	covy2 = covy2 / _nOrient * 2;
	covxy = 4 * covxy / _nOrient;

	Mat denom;
	magnitude(covxy, covx2 - covy2, denom);
	denom = denom + _epsilon;

	Mat M = (covy2 + covx2 + denom) / 2;
	Mat m = (covy2 + covx2 - denom) / 2;

	_M = M;
	_m = m;

	outEdges = M;
	outCorners = m;
}

void PhaseCongruency::Initialize(Size size)
{
	_size = size;
	int rows = size.height;
	int cols = size.width;

	// Set up X and Y matrices with ranges normalised to +/- 0.5
	vector<double> xRange(cols, 0);
	vector<double> yRange(rows, 0);
	Helper(cols, xRange);
	Helper(rows, yRange);

	// [x,y] = meshgrid(xrange, yrange);
	// radius = sqrt(x.^2 + y.^2);       
	// theta = atan2(-y, x);
	Mat radius = Mat::zeros(rows, cols, ELEM_TYPE);
	Mat theta = Mat::zeros(rows, cols, ELEM_TYPE);
	for (int r = 0; r < rows; r++)
	{
		double* ptrRadius = radius.ptr<double>(r);
		double* ptrTheta = theta.ptr<double>(r);
		for (int c = 0; c < cols; c++)
		{
			double x = xRange[c];
			double y = yRange[r];
			ptrRadius[c] = sqrt(x * x + y * y);
			ptrTheta[c] = atan2(-y, x);
		}
	}
	
	IFFTShift(radius);    // radius = ifftshift(radius);
	IFFTShift(theta);    // theta  = ifftshift(theta); 
	radius.at<double>(0, 0) = 1;

	Mat lp = LowPassFilter(size, 0.45, 15);
	std::vector<Mat> logGabor(_nScale);
	for (int s = 0; s < _nScale; s++)
	{
		double waveLen = _minWavelength * pow(_mult, s);
		double fo = 1.0 / waveLen;
		log(radius / fo, logGabor[s]);
		pow(logGabor[s], 2, logGabor[s]);
		exp(-logGabor[s] / (2 *  pow(log(_sigmaOnf), 2)), logGabor[s]);
		multiply(logGabor[s], lp, logGabor[s]);
		logGabor[s].at<double>(0, 0) = 0;
	}

	Mat matArr[2];
	matArr[0] = Mat::zeros(rows, cols, ELEM_TYPE);
	matArr[1] = Mat::zeros(rows, cols, ELEM_TYPE);
	Mat spread = Mat::zeros(rows, cols, ELEM_TYPE);
	for (int o = 0; o < _nOrient; o++)
	{
		double angle = o * M_PI / _nOrient;
		for (int r = 0; r < rows; r++)
		{
			double* ptr = theta.ptr<double>(r);
			double* ptrSpread = spread.ptr<double>(r);
			for (int j = 0; j < cols; j++)
			{
				double sinTheta = sin(ptr[j]);
				double cosTheta = cos(ptr[j]);
				double ds = sinTheta * cos(angle) - cosTheta * sin(angle);
				double dc = cosTheta * cos(angle) + sinTheta * sin(angle);
				double dTheta = abs(atan2(ds, dc));
				dTheta = min(dTheta * _nOrient / 2, M_PI);

				ptrSpread[j] = (cos(dTheta) + 1) * 0.5;
			}
		}

		for (int s = 0; s < _nScale; s++)
		{
			matArr[0] = logGabor[s].mul(spread);
			merge(matArr, 2, _filter[o * _nScale + s]);
		}
	}
}
