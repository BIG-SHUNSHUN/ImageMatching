#include "wavelet.h"

shun::WaveletPyramid::WaveletPyramid(Mat src, int layers, string name)
	:_bottom(src), _layers(layers)
{
	InitializeByWavelib(src, name);
}

shun::WaveletPyramid::~WaveletPyramid()
{
	for (int i = 0; i < _wtObjects.size(); i++)
	{
		wave_free(_wvObjects[i]);
		wt2_free(_wtObjects[i]);
		free(_wtCoeffs[i]);
	}
}

void shun::WaveletPyramid::ShowImage(int layerIdx, const char * type)
{
	if (layerIdx < 0 || layerIdx > _layers - 1)
	{
		cerr << "layerIdx is out of bound" << endl;
		return;
	}
		
	if (layerIdx == 0)
	{
		imshow("layer " + layerIdx, _bottom);
	}
	else
	{
		Mat img = GetCoeffs(layerIdx, type);
		normalize(img, img, 1, 0, NORM_MINMAX);
		imshow("layer " + layerIdx, img);
	}

	waitKey(0);
	destroyWindow("layer " + layerIdx);
}

Mat shun::WaveletPyramid::GetCoeffs(int layerIdx, const char * type)
{
	if (layerIdx < 1 || layerIdx > _layers - 1)
	{
		cerr << "layerIdx is out of bound" << endl;
		return Mat();
	}
	int rows = 0, cols = 0;
	double* coeff = getWT2Coeffs(_wtObjects[layerIdx - 1], _wtCoeffs[layerIdx - 1], 1, const_cast<char*>(type), &rows, &cols);
	Mat img(rows, cols, CV_64F, coeff);
	return img;
}

void shun::WaveletPyramid::InitializeByWavelib(Mat src, string name)
{
	// 原始影像的长宽
	int rows = src.rows;
	int cols = src.cols;

	// 将原始影像变为一维double数组
	Mat dst;
	src.convertTo(dst, CV_64FC1);
	vector<double> vecSrc = static_cast<vector<double>>(dst.reshape(1, 1));
	double* in = vecSrc.data();

	for (int i = 0; i < _layers - 1; i++)
	{
		// 小波变换器
		wave_object obj = wave_init(name.c_str());// Initialize the wavelet
		wt2_object wt = wt2_init(obj, "dwt", rows, cols, 1);
		_wvObjects.push_back(obj);
		_wtObjects.push_back(wt);

		// 变换结果
		double* waveCoeffs = dwt2(wt, in);
		_wtCoeffs.push_back(waveCoeffs);

		rows = wt->dimensions[0];
		cols = wt->dimensions[1];
		in = waveCoeffs;
	}
}

double absmax(double *array, int N) {
	double max;
	int i;

	max = 0.0;
	for (i = 0; i < N; ++i) {
		if (fabs(array[i]) >= max) {
			max = fabs(array[i]);
		}
	}

	return max;
}

double generate_rnd() {
	double rnd;

	rnd = (double)(rand() % 100 + 1);

	return rnd;
}

void shun::WavelibDemo()
{
	// 小波变换器
	const char *name = "db2";
	int rows = 10000, cols = 5000, J = 3;
	wave_object obj = wave_init(name);// Initialize the wavelet
	wt2_object wt = wt2_init(obj, "dwt", rows, cols, J);

	// 输入数据
	int N = rows * cols;
	double* inp = (double*)calloc(N, sizeof(double));
	srand(time(0));
	for (int i = 0; i < rows; ++i) {
		for (int k = 0; k < cols; ++k) {
			inp[i*cols + k] = generate_rnd();
		}
	}

	// 小波正变换
	double* wavecoeffs = dwt2(wt, inp);

	// 获取变换结果
	int ir, ic;
	double* cLL = getWT2Coeffs(wt, wavecoeffs, 1, const_cast<char*>("D"), &ir, &ic);
	dispWT2Coeffs(cLL, ir, ic);

	// 逆变换
	double* oup = (double*)calloc(N, sizeof(double));
	idwt2(wt, wavecoeffs, oup);

	// 与源数据的差异
	double* diff = (double*)calloc(N, sizeof(double));
	for (int i = 0; i < rows*cols; ++i) {
		diff[i] = oup[i] - inp[i];
	}
	double amax = absmax(diff, rows*cols);
	printf("\nAbs Max %g \n", amax);

	// 小波变换的参数
	wt2_summary(wt);

	wave_free(obj);
	wt2_free(wt);
	free(inp);
	free(wavecoeffs);
	free(oup);
	free(diff);
}

void shun::WaveletDecomposition(Mat img, int layers)
{
	// 定义haar尺度函数和小波函数
	float h_f[] = { 0.7071, 0.7071 };    //1/squr(2)
	float g_f[] = { 0.7071f, -0.7071f };    //wavelet filter

	// 分配临时空间
	int rows = img.rows, cols = img.cols;
	vector<vector<double>> buffer(rows, vector<double>(cols, 0));
	//double** buffer = new double*[rows];
	//for (int i = 0; i < rows; i++)
	//{
	//	buffer[i] = new double[cols];
	//}

	// 小波分解
	for (int layer = 0; layer < layers - 1; layer++)
	{
		int h = rows / 2;
		int w = cols / 2;
		for (int i = 0; i < rows; i = i + 1)
		{
			uchar* ptr = img.data + img.step * i;
			for (int j = 0; j < cols; j = j + 2)
			{
				int c = j / 2;
				buffer[i][c] = (ptr[j] * h_f[0] + ptr[j + 1] * h_f[1]);
				buffer[i][c + w] = (ptr[j] * g_f[0] + ptr[j + 1] * g_f[1]);
			}
		}
		for (int j = 0; j < w; j = j + 1)
		{
			for (int i = 0; i < rows; i = i + 2)
			{
				int r = i / 2;
				uchar* ptr1 = img.data + img.step * r;
				uchar* ptr2 = img.data + img.step * (r + h);
				ptr1[j] = unsigned char((buffer[i][j] * h_f[0] + buffer[i + 1][j] * h_f[1]) / 2);    //a
				ptr2[j] = unsigned char(((buffer[i][j + w] * h_f[0] + buffer[i + 1][j + w] * h_f[1]) + 255) / 2);    //v
				ptr1[j + w] = unsigned char(((buffer[i][j] * g_f[0] + buffer[i + 1][j] * g_f[1]) + 255) / 2);    //l
				ptr2[j + w] = unsigned char(((buffer[i][j + w] * g_f[0] + buffer[i + 1][j + w] * g_f[1]) + 255) / 2);    //X
			}
		}
		rows = h;
		cols = w;
	}
}

void shun::WaveletRecomstruction(Mat img, int layers, int curLayer, double highFreqFactor)
{
	// 如果是顶层，不重建
	if (curLayer == 1)
		return;

	// 尺度函数和小波函数
	float h_f[] = { 0.7071f, 0.7071f };    //1/squr(2)
	float g_f[] = { 0.7071f, -0.7071f };    //wavelet filter

	// 当前层的影像大小
	int rows = img.rows / pow(2, (layers - 1)) * pow(2, curLayer - 1);
	int cols = img.cols / pow(2, (layers - 1)) * pow(2, curLayer - 1);

	vector<vector<double>> buffer(rows, vector<double>(cols, 0));

	// 上一层的影像大小
	int h = rows / 2;
	int w = cols / 2;

	// 小波重建
	float g1[2][2], g2[2][2], g3[2][2], g4[2][2];
	for (int i = 0; i < h; i++)
	{
		uchar* ptr_i = img.data + img.step * i;
		uchar* ptr_ih = img.data + img.step * (i + h);
		for (int j = 0; j < w; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				g1[0][k] = ptr_i[j] * h_f[k] * 2;
			}
			for (int k = 0; k < 2; k++)
			{
				for (int l = 1; l >= 0; l--)
				{
					g1[l][k] = g1[0][k] * h_f[l];
				}
			}
			for (int k = 0; k < 2; k++)
			{
				g2[0][k] = (ptr_i[j + w] * 2 - 254)*h_f[k] * highFreqFactor;
			}
			for (int k = 0; k < 2; k++)
			{
				for (int l = 1; l >= 0; l--)
				{
					g2[l][k] = g2[0][k] * g_f[l];
				}
			}
			for (int k = 0; k < 2; k++)
			{
				g3[0][k] = (ptr_ih[j] * 2 - 254)*g_f[k] * highFreqFactor;
			}
			for (int k = 0; k < 2; k++)
			{
				for (int l = 1; l >= 0; l--)
				{
					g3[l][k] = g3[0][k] * h_f[l];
				}
			}
			for (int k = 0; k < 2; k++)
			{
				g4[0][k] = (ptr_ih[j + w] * 2 - 254)*g_f[k] * highFreqFactor;
			}
			for (int k = 0; k < 2; k++)
			{
				for (int l = 1; l >= 0; l--)
				{
					g4[l][k] = g4[0][k] * g_f[l];
				}
			}
			for (int l = 0; l < 2; l++)
			{
				for (int k = 0; k < 2; k++)
				{
					int r = i * 2 + l;
					int c = j * 2 + k;
					buffer[r][c] = g1[l][k] + g2[l][k] + g3[l][k] + g4[l][k];
				}
			}
		}
	}
	for (int i = 0; i < rows; i++)
	{
		uchar* ptr = img.data + img.step * i;
		for (int j = 0; j < cols; j++)
		{
			if (buffer[i][j] > 255)
			{
				ptr[j] = 255;
			}
			else if (buffer[i][j] < 0)
			{
				ptr[j] = 0;
			}
			else
			{
				ptr[j] = unsigned char(buffer[i][j]);
			}
		}
	}
}

void se(vector<vector<float>>& a, int n, vector<float>& r, int m)
{
	float b, ab1;
	int i, j, k, j1, n1, n2, n3, l;

	for (i = 0; i < n; i++)
	{
		r[i] = 0.f;
		b = 0.f;
		for (j = i; j < n; j++)
		{
			if (fabs(b) <= fabs(a[j][i]))

			{
				b = a[j][i];
				j1 = j;
			}
			else
				continue;
		}
		for (k = i; k < m; k++)
		{
			ab1 = a[j1][k] / b;
			a[j1][k] = a[i][k];
			a[i][k] = ab1;
		}
		n1 = i + 1;
		if (n1 < n)
		{
			for (j = n1; j < n; j++)
			{
				for (k = n1; k < m; k++)
				{
					a[j][k] = a[j][k] - a[j][i] * a[i][k];
				}

			}
		}
	}
	r[n - 1] = a[n - 1][m - 1];
	n2 = n - 1;
	for (i = 0; i < n2; i++)
	{
		l = n2 - 1 - i;
		r[l] = a[l][m - 1];
		n3 = l + 1;
		for (j = n3; j < n; j++)
		{
			r[l] = r[l] - r[j] * a[l][j];
		}
	}
}

void shun::LeastSquareMatch(Mat imgL, Mat imgR, int wsizex, int wsizey, Point2f & ptL, Point2f & ptR)
{
	int iterationN;
	float sxr, syr, dx, dy;
	float cc;
	float absmin;

	int war;    //左窗口中总点数
	int  wlx0, wly0;    //左窗口中心坐标(即窗口一半处)
	war = wsizey * wsizex;    //左窗口总像点数
	wly0 = wsizey / 2;		wlx0 = wsizex / 2;

	vector<vector<float>> aard(war, vector<float>(3, 0));
	vector<vector<float>> ard(2, vector<float>(3, 0));
	vector<float> rrd(2, 0);
	vector<vector<float>> aa(war, vector<float>(7, 0));
	vector<vector<float>> a(6, vector<float>(7, 0));
	vector<float> r(6, 0);

	vector<vector<float>> wpl(wsizey, vector<float>(wsizex, 0));    //用于匹配计算的左窗口
	vector<vector<float>> patch(wsizey, vector<float>(wsizex, 0));    //用于匹配计算的右窗口
	vector<vector<float>> dg(wsizey, vector<float>(wsizex, 0));    //影像灰度差

	float clargest = 0.f;
	absmin = 99999.0f;
	iterationN = 0;

	float h0, h1, c0, c1, c2, c3, c4, c5;
	c1 = c4 = 1;
	c2 = c5 = 0;
	c0 = ptR.x - wlx0; c3 = ptR.y - wly0;

	int  wlxs, wlys, wlxe, wlye;
	wlxs = (int)ptL.x - wlx0;	wlxe = (int)ptL.x + wlx0 + 1;
	wlys = (int)ptL.y - wly0;	wlye = (int)ptL.y + wly0 + 1;

	/*****************get left windows image******************/
	for (int i = wlys; i < wlye; i++)
	{
		int k = i - wlys;
		for (int j = wlxs; j < wlxe; j++)
			wpl[k][j - wlxs] = (float)imgL.at<uchar>(i, j);
	}

	/************************ matching start ******************************/
	//左影像灰度平均值
	float avgl = 0.f;
	for (int ii = 0; ii < wsizey; ii++)
		for (int jj = 0; jj < wsizex; jj++)
			avgl = avgl + wpl[ii][jj];
	avgl = avgl / war;
	//左影像灰度平方总和
	float dltagl = 0.f;
	for (int ii = 0; ii < wsizey; ii++)
		for (int jj = 0; jj < wsizex; jj++)
			dltagl = dltagl + wpl[ii][jj] * wpl[ii][jj];
	dltagl = dltagl / war - avgl * avgl;

	while (true)
	{
		for (int ii = 0; ii < wsizey; ii++)
		{
			for (int jj = 0; jj < wsizex; jj++)
			{
				sxr = c0 + c1 * jj + c2 * ii;    // 重新计算右窗口中心在右影像中的位置
				syr = c3 + c4 * ii + c5 * jj;
				int l = int(sxr);
				int k = int(syr);
				dx = sxr - l;
				dy = syr - k;
				patch[ii][jj] = (1 - dx)*(1 - dy)*imgR.at<uchar>(k, l) +
					(1 - dy)*dx*imgR.at<uchar>(k, l + 1) +
					(1 - dx)*dy*imgR.at<uchar>(k + 1, l) +
					dx * dy*imgR.at<uchar>(k + 1, l + 1);
			}
		}
		/************************** radi0 correction **************************/
		int n = 2, m = 3;
		for (int ii = 0; ii < n; ii++)
			for (int jj = 0; jj < m; jj++)
				ard[ii][jj] = 0.0f;
		for (int ii = 0; ii < wsizey; ii++)
		{
			for (int jj = 0; jj < wsizex; jj++)
			{
				int k = ii * wsizex + jj;
				aard[k][0] = 1.f;
				aard[k][1] = patch[ii][jj];
				aard[k][2] = wpl[ii][jj];
			}
		}
		for (int ii = 0; ii < war; ii++)
		{
			for (int jj = 0; jj < n; jj++)
				for (int k = 0; k < m; k++)
					ard[jj][k] = ard[jj][k] + aard[ii][jj] * aard[ii][k];
		}
		se(ard, n, rrd, m);
		h0 = rrd[0]; h1 = rrd[1];
		for (int ii = 0; ii < wsizey; ii++)
			for (int jj = 0; jj < wsizex; jj++)
				patch[ii][jj] = patch[ii][jj] * h1 + h0;
		printf("%s %f %f\n", "radio corrections  ", h0, h1);
		/*********************  correlation conficience  **********************/
						//右影像灰度平均值（搜索窗口影像）
		float avgr = 0.f;
		for (int ii = 0; ii < wsizey; ii++)
			for (int jj = 0; jj < wsizex; jj++)
				avgr = avgr + patch[ii][jj];
		avgr = avgr / war;
		float dltagr = 0.f, dltaglr = 0.f, cgmabs = 0.f;
		for (int ii = 0; ii < wsizey; ii++)
		{
			for (int jj = 0; jj < wsizex; jj++)
			{
				dltagr = dltagr + patch[ii][jj] * patch[ii][jj];
				dltaglr = dltaglr + wpl[ii][jj] * patch[ii][jj];
				cgmabs = cgmabs + fabs((wpl[ii][jj] - avgl) - (patch[ii][jj] - avgr));
			}
		}
		dltagr = dltagr / war - avgr * avgr;
		dltaglr = dltaglr / war - avgl * avgr;
		if (dltagl <= 0 || dltagr <= 0)
			cc = 0.f;
		else
			cc = dltaglr / sqrt(dltagl*dltagr);
		if (cgmabs <= absmin)
			absmin = cgmabs;
		if (cc > clargest)
			clargest = cc;
		else
		{
			if (cc == clargest && cgmabs == absmin)
				clargest = cc;
		}

		/***************** Least square solution ******************************/
		n = 6; m = 7;
		for (int ii = 0; ii < n; ii++)
			for (int jj = 0; jj < m; jj++)
				a[ii][jj] = 0.0f;

		for (int ii = 0; ii < wsizey; ii++)
		{
			for (int jj = 0; jj < wsizex; jj++)
			{
				dg[ii][jj] = patch[ii][jj] - wpl[ii][jj];
				if (ii > 0)
					syr = patch[ii][jj] - patch[ii - 1][jj];//坡度
				else
					syr = patch[ii + 1][jj] - patch[ii][jj];
				if (jj > 0)
					sxr = patch[ii][jj] - patch[ii][jj - 1];
				else
					sxr = patch[ii][jj + 1] - patch[ii][jj];
				int k = ii * wsizex + jj;//点序
				//误差方程系数
				//V=-SXr*DX-SXr*jj*DFx-SXr*ii*DRx-SYr*DY-SYr*ii*DFy-SYr*jj*DRy+dg
				aa[k][0] = (-sxr);
				aa[k][1] = (-sxr)*jj;
				aa[k][2] = (-sxr)*ii;
				aa[k][3] = (-syr);
				aa[k][4] = (-syr)*ii;
				aa[k][5] = (-syr)*jj;
				aa[k][6] = dg[ii][jj];
			}
		}
		//组法方程
		for (int ii = 0; ii < war; ii++)
		{
			for (int jj = 0; jj < n; jj++)
				for (int k = 0; k < m; k++)
					a[jj][k] = a[jj][k] + aa[ii][jj] * aa[ii][k];
		}
		se(a, n, r, m);
		c0 = c0 + r[0];
		c1 = c1 + r[1];
		c2 = c2 + r[2];
		c3 = c3 + r[3];
		c4 = c4 + r[4];
		c5 = c5 + r[5];
		dx = r[0] + r[1] * wlx0 + r[2] * wly0;
		dy = r[3] + r[4] * wly0 + r[5] * wlx0;
		printf("%s %f %f \n", "dy dx", dy, dx);
		if (cc == clargest && iterationN < 10)
		{
			ptR.x = ptR.x + dx;
			ptR.y = ptR.y + dy;
			iterationN = iterationN + 1;
		}
		else
		{
			break;
		}
	}
}
