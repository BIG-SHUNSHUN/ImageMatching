#include "phase.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <corecrt_math_defines.h>
#include "utils.h"

using namespace cv;
using namespace std;
using namespace shun;

void shun::PhaseCongruency::Helper(int n, vector<double>& arr)
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

Mat shun::PhaseCongruency::LowPassFilter(Size size, double cutOff, int n)
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
shun::PhaseCongruency::PhaseCongruency()
{
}

//Phase congruency calculation
void shun::PhaseCongruency::Prepare(Mat src)
{
	Size size = src.size();
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
	std::vector<Mat> logGabor(_params.nScale);
	for (int s = 0; s < _params.nScale; s++)
	{
		double waveLen = _params.minWavelength * pow(_params.mult, s);
		double fo = 1.0 / waveLen;
		log(radius / fo, logGabor[s]);
		pow(logGabor[s], 2, logGabor[s]);
		exp(-logGabor[s] / (2 * pow(log(_params.sigmaOnf), 2)), logGabor[s]);
		multiply(logGabor[s], lp, logGabor[s]);
		logGabor[s].at<double>(0, 0) = 0;
	}

	Mat matArr[2];
	matArr[0] = Mat::zeros(rows, cols, ELEM_TYPE);
	matArr[1] = Mat::zeros(rows, cols, ELEM_TYPE);
	Mat spread = Mat::zeros(rows, cols, ELEM_TYPE);
	vector<Mat> filter(_params.nScale * _params.nOrient);
	for (int o = 0; o < _params.nOrient; o++)
	{
		double angle = o * M_PI / _params.nOrient;
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
				dTheta = min(dTheta * _params.nOrient / 2, M_PI);

				ptrSpread[j] = (cos(dTheta) + 1) * 0.5;
			}
		}

		for (int s = 0; s < _params.nScale; s++)
		{
			matArr[0] = logGabor[s].mul(spread);
			merge(matArr, 2, filter[o * _params.nScale + s]);
		}
	}

	// Ô­Í¼¸µÀïÒ¶±ä»»
	Mat planes[] = { Mat_<double>(src), Mat::zeros(src.size(), ELEM_TYPE) };
	Mat imageFFT;
	merge(planes, 2, imageFFT);        
	dft(imageFFT, imageFFT);            

	Mat pcSum = Mat::zeros(src.size(), ELEM_TYPE);
	Mat An;
	Mat sorted;
	Mat maxAn = Mat::zeros(src.size(), ELEM_TYPE);
	Mat sumE = Mat::zeros(src.size(), ELEM_TYPE);
	Mat sumO = Mat::zeros(src.size(), ELEM_TYPE);
	Mat sumAn = Mat::zeros(src.size(), ELEM_TYPE);
	Mat energy = Mat::zeros(src.size(), ELEM_TYPE);
	Mat eo;
	Mat xEnergy, meanE, meanO;
    for (int o = 0; o < _params.nOrient; o++)
    {
		sumE.setTo(0);
		sumO.setTo(0);
		sumAn.setTo(0);
		energy.setTo(0);
		double tau = 0;
		
        for (int s = 0; s < _params.nScale; s++)
        {
            mulSpectrums(imageFFT, filter[o * _params.nScale + s], eo, 0);    // Convolution
			idft(eo, eo, DFT_SCALE);

			Mat complex[2];
			split(eo, complex);
			_eo[s][o].e = complex[0];
			_eo[s][o].o = complex[1];
            magnitude(complex[0], complex[1], An);

			sumAn += An;
			sumE += complex[0];
			sumO += complex[1];

			if (s == 0)
			{
				if (_params.noiseMethod == -1)
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

		magnitude(sumE, sumO, xEnergy); 
		xEnergy += DBL_EPSILON;
		divide(sumE, xEnergy, meanE);
		divide(sumO, xEnergy, meanO);

		for (int s = 0; s < _params.nScale; s++)
		{
			Mat tempE = _eo[s][o].e;
			Mat tempO = _eo[s][o].o;
			energy += tempE.mul(meanE) + tempO.mul(meanO) -
				abs(tempE.mul(meanO) - tempO.mul(meanE));
		}

        double T = 0;
		if (_params.noiseMethod >= 0)
		{
			T = _params.noiseMethod;
		}
		else
		{
			double totalTau = tau * (1 - pow(1 / _params.mult, _params.nScale)) / (1 - (1 / _params.mult));
			double noiseMean = totalTau * sqrt(M_PI / 2);
			double noiseSigma = totalTau * sqrt((4 - M_PI) / 2);
			T = noiseMean + _params.k * noiseSigma;
		}

        max(energy -= T, 0.0, energy);

		Mat width = (sumAn / (maxAn + DBL_EPSILON) - 1.0) / (_params.nScale - 1.0);
		//cv::divide(sumAn, maxAn + _epsilon, width);
		//width = (width - 1) / (_nScale - 1.0);

		Mat weight;
		exp((_params.cutOff - width) * _params.g, weight);
		weight = 1.0 / (1.0 + weight);

		cv::divide(weight.mul(energy), sumAn, _pc[o]);

		pcSum += _pc[o];
    }//orientation
	_pcSum = pcSum;
}

void shun::PhaseCongruency::Feature(cv::Mat& outEdges, cv::Mat& outCorners)
{
	Mat covx2 = Mat::zeros(_size, ELEM_TYPE);
	Mat covy2 = Mat::zeros(_size, ELEM_TYPE);
	Mat covxy = Mat::zeros(_size, ELEM_TYPE);
	for (int o = 0; o < _params.nOrient; o++)
	{
		double angle = o * M_PI / _params.nOrient;
		Mat tmpX = _pc[o] * cos(angle);
		Mat tmpY = _pc[o] * sin(angle);

		covx2 += tmpX.mul(tmpX);
		covy2 += tmpY.mul(tmpY);
		covxy += tmpX.mul(tmpY);
	}

	covx2 = covx2 / _params.nOrient * 2;
	covy2 = covy2 / _params.nOrient * 2;
	covxy = 4 * covxy / _params.nOrient;

	Mat denom;
	magnitude(covxy, covx2 - covy2, denom);
	denom = denom + DBL_EPSILON;

	Mat M = (covy2 + covx2 + denom) / 2;
	Mat m = (covy2 + covx2 - denom) / 2;

	outEdges = M;
	outCorners = m;
}

void shun::PhaseCongruency::Orientation(cv::Mat & orient)
{
	Mat EnergyV[2] = { Mat::zeros(_size, ELEM_TYPE) , Mat::zeros(_size, ELEM_TYPE) };
	Mat sumO = Mat::zeros(_size, ELEM_TYPE);
	for (int o = 0; o < _params.nOrient; o++)
	{
		double theta = o * M_PI / _params.nOrient;
		sumO.setTo(0);
		for (int s = 0; s < _params.nScale; s++)
		{
			Mat tempO = _eo[s][o].o;
			sumO += tempO;
		}
		EnergyV[0] += sumO * cos(theta);
		EnergyV[1] += sumO * sin(theta);
	}

	int rows = _size.height;
	int cols = _size.width;
	orient.create(_size, ELEM_TYPE);
	for (int r = 0; r < rows; r++)
	{
		double* ptr0 = EnergyV[0].ptr<double>(r);
		double* ptr1 = EnergyV[1].ptr<double>(r);
		double* ptrOri = orient.ptr<double>(r);
		for (int c = 0; c < cols; c++)
		{
			ptrOri[c] = atan2(ptr1[c], ptr0[c] + 0.000001);
		}
	}
}

const PhaseCongruencyParams & shun::PhaseCongruency::Params()
{
	return _params;
}

void shun::PhaseCongruency::SetParams(const PhaseCongruencyParams& params)
{
	_params = params;

	_eo.resize(params.nScale, vector<EO>(params.nOrient));
	_pc.resize(params.nOrient);
}

void shun::PhaseCongruency::SetParams(int nScale, int nOrient, double minWavelength, double mult, 
	                                  double sigmaOnf, double k, double cutoff, double g, double noiseMethod)
{
	_params.nOrient = nOrient;
	_params.nScale = nScale;
	_params.minWavelength = minWavelength;
	_params.mult = mult;
	_params.sigmaOnf = sigmaOnf;
	_params.k = k;
	_params.cutOff = cutoff;
	_params.g = g;
	_params.noiseMethod = noiseMethod;

	_eo.resize(nScale, vector<EO>(nOrient));
	_pc.resize(nOrient);
}
