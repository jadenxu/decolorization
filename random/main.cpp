//-----------------------------------------------------------------
//opencv.cpp is for practice opencv by myself and purpose of urop
//-----------------------------------------------------------------

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<math.h>
#include<Eigen/dense>
#include<cstdlib>
#include<ctime>
using namespace Eigen;
using namespace std;
using namespace cv;

const double PI = 3.14159265;

double carmen_gaussian_random(double mean, double std)
{
  const double norm = 1.0 / (RAND_MAX + 1.0);
  double u = 1.0 - rand() * norm;                  /**//* can't let u == 0 */
  double v = rand() * norm;
  double z = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
  return mean + std * z;
} 


/* The useless code (convert rgb to lab, writen by myself)

Mat rgb2xyz(Mat& image)
{
	//for D65
	//[ X ]   [  0.412453  0.357580  0.180423 ]   [ R ] **
	//[ Y ] = [  0.212671  0.715160  0.072169 ] * [ G ]
	//[ Z ]   [  0.019334  0.119193  0.950227 ]   [ B ].
	Mat xyz(image.rows, image.cols, CV_32FC3);

	for(int i = 0; i < image.rows; i++)
	{
		for(int j = 0; j < image.cols; j++)
		{
			xyz.at<Vec3f>(i,j)[0] = 0.412453 * ((float)image.at<Vec3b>(i,j)[2]/255.0) + 0.357580 * ((float)image.at<Vec3b>(i,j)[1]/255.0) + 0.180423 * ((float)image.at<Vec3b>(i,j)[0]/255.0);
			xyz.at<Vec3f>(i,j)[1] = 0.212671 * ((float)image.at<Vec3b>(i,j)[2]/255.0) + 0.715160 * ((float)image.at<Vec3b>(i,j)[1]/255.0) + 0.072169 * ((float)image.at<Vec3b>(i,j)[0]/255.0);
			xyz.at<Vec3f>(i,j)[2] = 0.019334 * ((float)image.at<Vec3b>(i,j)[2]/255.0) + 0.119193 * ((float)image.at<Vec3b>(i,j)[1]/255.0) + 0.950227 * ((float)image.at<Vec3b>(i,j)[0]/255.0);
		}
	}

	return xyz;
}

double f_xyz_lab(double t )
{
	if(t > pow((6.0 / 29.0),3))
	{
		return pow(t,1.0 / 3.0);
	}
	else
	{
		return (1.0 / 3.0) * pow(29.0 / 6.0,2) * t + (4.0 / 29.0);
	}
}

Mat rgb2lab(Mat& image)
{
	//we first transfer it to xyz
	Mat lab = rgb2xyz(image);

	//then we transfer xyz to lab
	//D65 95.0182 100.0000 108.7485

	double d65[3] = {0.950182,1.0,1.087485};
	double l,a,b;
	for(int i = 0; i < image.rows; i++)
	{
		for(int j =0; j < image.cols; j++)
		{
			l = 116.0 * f_xyz_lab(lab.at<Vec3f>(i,j)[1] / d65[1]) - 16.0;
			a = 500.0 * (f_xyz_lab(lab.at<Vec3f>(i,j)[0] / d65[0])- f_xyz_lab(lab.at<Vec3f>(i,j)[1] / d65[1]));
			b = 200.0 * (f_xyz_lab(lab.at<Vec3f>(i,j)[1] / d65[1])- f_xyz_lab(lab.at<Vec3f>(i,j)[2] / d65[2]));
			lab.at<Vec3f>(i,j)[0] = l;
			lab.at<Vec3f>(i,j)[1] = a;
			lab.at<Vec3f>(i,j)[2] = b;
		}
	}

	return lab;
}*/

Mat lab2lch(Mat& image)
{
	Mat lch(image.rows, image.cols, CV_32FC3);
	for(int i = 0; i < image.rows; i++)
	{
		for(int j = 0; j < image.cols; j++)
		{
			//calculate c and h (the range of a, b in opencv is (0,255), we bring it back to (-128,127) by minus 128)
			double c = sqrt((float)((image.at<Vec3b>(i,j)[1] - 128) * (image.at<Vec3b>(i,j)[1] - 128) + (image.at<Vec3b>(i,j)[2] - 128) * (image.at<Vec3b>(i,j)[2] - 128)));
			double h = atan2((float)(image.at<Vec3b>(i,j)[2] - 128), (float)(image.at<Vec3b>(i,j)[1] - 128));
			
			//keep h be in suitable range
			if(h < 0)
				h = h + 2 * PI;
			while(h > 2 * PI)
				h = h - 2 * PI;

			//cout<<h<<endl;
			//assign value to lch
			lch.at<Vec3f>(i,j)[0] = image.at<Vec3b>(i,j)[0] / 255.0 * 100.0;
			lch.at<Vec3f>(i,j)[1] = c; // 255.0 * 100.0; 
			lch.at<Vec3f>(i,j)[2] = h;
		}
	}

	return lch;
}

double H_KEffect(Vec3f m)
{
	double hkl;
	//D65 95.0182 100.0000 108.7485
	
	//the range of u and v in opencv is between (0, 255), bring in back to (-134, 220) and (-140,122) 
	double minus_u = m[1] / 255.0 * 354.0 - 134.0;
	double minus_v = m[2] / 255.0 * 256.0 - 140.0;

	//get u' - u'c, v' - v'c
	double tem = m[0] / 255.0 * 100.0;
	minus_u = minus_u / (13.0 * tem);
	minus_v = minus_v / (13.0 * tem);

	//calculate the theta, s, q and k (for k we use 20 for La)
	double theta = atan2(minus_v, minus_u);
	double s = 13 * sqrt(minus_u * minus_u + minus_v * minus_v);
	double q = -0.01585 - 0.03017 * cos(theta) - 0.04556 * cos(2 * theta) - 0.02667 * cos(3 * theta) - 0.00295 * cos(4 * theta) 
		        + 0.14592 * sin(theta) + 0.05084 * sin(2 * theta) - 0.01900 * sin(3 * theta) - 0.00764 * sin(4 * theta);
	double k = (6.469 + 6.362 * pow(20.0, 0.4495)) / (6.469 + pow(20.0, 0.4495));
	k = k * 0.2717;

	//calculate hkl
	hkl = tem + (-0.1340 * q + 0.0872 * k) * s * tem;
	hkl = hkl /100.0 * 255;

	return hkl;
}

double gradient_color(Vec3f i, Vec3f j, Vec3f m, Vec3f n)
{
	
	double delta_l = (i[0] - j[0]) / 255.0 * 100.0;
	double delta_a = i[1] - j[1];
	double delta_b = i[2] - j[2];
	int sign = 0; 

	//calculate the HK_Effect for these two pixels in Luv
	double hkl1,hkl2;
	hkl1 = H_KEffect(m);
	hkl2 = H_KEffect(n);
	
	//the priority for sign function
	if(hkl1 - hkl2 != 0)
	{
		if(hkl1 - hkl2 > 0)
			sign = 1;
		else
			sign = -1;
	}
	else if(delta_l != 0)
	{
		if(delta_l > 0)
			sign = 1;
		else
			sign = -1;
	}
	else
	{
		if(pow(delta_l,3) + pow(delta_a,3) + pow(delta_b,3) > 0)
			sign = 1;
		else
			sign = -1;
	}

	//calculate the gradient of these two pixels in Lab
	double result = sqrt(delta_a * delta_a + delta_b * delta_b);
	result = result / (2.54 * sqrt(2.0));
	result = result * result + delta_l * delta_l;
	result = sqrt(result);
	result = sign * result;
	
	return result;
}

void color2grayscale(Mat& image)
{
	//first we get the CIE LAB, CIE LCH, CIE LUV for the following use
	Mat lab,lch,luv;
	cvtColor(image, lab, CV_BGR2Lab);
	cvtColor(image, luv, CV_BGR2Luv);
	lch = lab2lch(lab);

	/*
	Mat ok(1,1,CV_8UC3), lab1, lch1, luv1;
	ok.at<Vec3b>(0,0)[0] = 255;
	ok.at<Vec3b>(0,0)[1] = 0;
	ok.at<Vec3b>(0,0)[2] = 0;
	cvtColor(ok, lab1, CV_BGR2Lab);
	cvtColor(ok, luv1, CV_BGR2Luv);
	lch1 = lab2lch(lab1);
	cout<<"LAB "<< lab1.at<Vec3b>(0,0)[0] / 255.0 * 100.0 <<" "<<lab1.at<Vec3b>(0,0)[1] - 128 <<" "<<lab1.at<Vec3b>(0,0)[2] - 128<<endl;
	cout<<"LCH "<< lch1.at<Vec3f>(0,0)[0] / 255.0 * 100.0 <<" "<<lch1.at<Vec3f>(0,0)[1] <<" "<<lch1.at<Vec3f>(0,0)[2] * (180.0 / PI)<<endl;
	cout<<"LUV "<< luv1.at<Vec3b>(0,0)[0] / 255.0 * 100.0 <<" "<<luv1.at<Vec3b>(0,0)[1] / 255.0 * 354.0 - 134.0<<" "<<luv1.at<Vec3b>(0,0)[2] / 255.0 * 256.0 - 140.0<<endl;
	//----------------------------------------------------------------------------------------------------
	//DEBUG
	//the mat of gradient in color image, gradient in grayscale image,difference between these two gradient, HK_Effect
	//-----------------------------------------------------------------------------------------------------
	Mat gradientC;
	image.copyTo(gradientC);
	Mat gradientG;
	image.copyTo(gradientG);
	Mat gradient;
	image.copyTo(gradient);
	Mat HK_Effect;
	image.copyTo(HK_Effect);
	*/


	//-----------------------------------------------
	//MY
	//The random algorithm
	//-----------------------------------------------
	const int sigma = (lch.rows + lch.cols) / 20.0 ;
	double w[9], wv[9], wm[9][9], tem1[9], tem2[9];
	for(int i = 0; i < 9; i++)
	{
		w[i] = wv[i] = 0;
		for(int j = 0; j < 9; j++)
		{
			wm[i][j] = 0;
		}
	}

	for(int i = 0; i < lch.rows; i++)
	{
		for(int j = 0; j < lch.cols; j++)
		{
			int x = carmen_gaussian_random(0, 2 / PI * sigma * sigma);
			int y = carmen_gaussian_random(0, 2 / PI * sigma * sigma);
			while(!(x + i > 0 && x + i < lch.rows && y + j > 0 && y + j < lch.cols))
			{
				x = carmen_gaussian_random(0, 2 / PI * sigma * sigma);
				y = carmen_gaussian_random(0, 2 / PI * sigma * sigma);
			}
			//cout<< x <<" "<< y << endl; 

			int new_x = x + i;
			int new_y = y + j;
			
			for(int k = 0; k < 4; k++)
				tem1[k] = cos((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
			for(int k = 0; k < 4; k++)
				tem1[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
			tem1[8] = lch.at<Vec3f>(i,j)[1];

			for(int k = 0; k < 4; k++)
				tem2[k] = cos((k+1) * lch.at<Vec3f>(new_x,new_y)[2]) * lch.at<Vec3f>(new_x,new_y)[1];
			for(int k = 0; k < 4; k++)
				tem2[4 + k] = sin((k+1) * lch.at<Vec3f>(new_x,new_y)[2]) * lch.at<Vec3f>(new_x,new_y)[1];
			tem2[8] = lch.at<Vec3f>(new_x,new_y)[1];

			for(int k = 0; k < 9; k++)
				w[k] = tem1[k] - tem2[k];

			int r = gradient_color(lab.at<Vec3b>(i,j), lab.at<Vec3b>(new_x, new_y), luv.at<Vec3b>(i,j), luv.at<Vec3b>(new_x,new_y)) - (lch.at<Vec3f>(i,j)[0] - lch.at<Vec3f>(new_x,new_y)[0]);
			
			for(int m = 0; m < 9; m++)
			{
				for(int n = 0; n < 9; n++)
				{
					wm[m][n] += w[m] * w[n];
				}
			}

			for(int k = 0; k < 9; k++)
				wv[k] += r * w[k];
		}
	}


	//initialize matrix Ms and vectors Bs
	double ms[9][9];
	double bs[9], bs1[9];
	for(int i = 0; i < 9; i++)
	{
		bs[i] = bs1[i] = 0.0;
		for(int j = 0; j < 9; j++)
			ms[i][j] = 0.0;
	}

	//----------------------------------
	//calculate Ms and	Bs, and we assume the number of freedom is 4
	//----------------------------------
	double u[9], v[9], p, q;
	double t1[9], t2[9];
	double test = 0.0;
	for(int i = 0; i < lch.rows; i++)
	{
		for(int j = 0; j < lch.cols; j++)
		{
			//--------------------------------
			//calculate u
			//--------------------------------
			if(i == 0)
			{
				for(int k = 0 ; k < 4; k++)
					t1[k] = cos((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				for(int k = 0; k < 4; k++)
					t1[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				t1[8] = lch.at<Vec3f>(i,j)[1];

				for(int k = 0 ; k < 4; k++)
					t2[k] = cos((k+1) * lch.at<Vec3f>(i+1,j)[2]) * lch.at<Vec3f>(i+1,j)[1];
				for(int k = 0; k < 4; k++)
					t2[4 + k] = sin((k+1) * lch.at<Vec3f>(i+1,j)[2]) * lch.at<Vec3f>(i+1,j)[1];
				t2[8] = lch.at<Vec3f>(i+1,j)[1];
				for(int k = 0; k < 9; k++)
				{
					u[k] = t2[k] - t1[k];
				}
				//cout<<endl;
			}
			else if(i == lch.rows - 1)
			{
				for(int k = 0 ; k < 4; k++)
					t1[k] = cos((k+1) * lch.at<Vec3f>(i - 1,j)[2]) * lch.at<Vec3f>(i - 1,j)[1];
				for(int k = 0; k < 4; k++)
					t1[4 + k] = sin((k+1) * lch.at<Vec3f>(i - 1,j)[2]) * lch.at<Vec3f>(i - 1,j)[1];
				t1[8] = lch.at<Vec3f>(i - 1,j)[1];

				for(int k = 0 ; k < 4; k++)
					t2[k] = cos((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				for(int k = 0; k < 4; k++)
					t2[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				t2[8] = lch.at<Vec3f>(i,j)[1];
				for(int k = 0; k < 9; k++)
					u[k] = t2[k] - t1[k];
			}
			else
			{
				for(int k = 0 ; k < 4; k++)
					t1[k] = cos((k+1) * lch.at<Vec3f>(i-1,j)[2]) * lch.at<Vec3f>(i-1,j)[1];
				for(int k = 0; k < 4; k++)
					t1[4 + k] = sin((k+1) * lch.at<Vec3f>(i-1,j)[2]) * lch.at<Vec3f>(i-1,j)[1];
				t1[8] = lch.at<Vec3f>(i-1,j)[1];

				for(int k = 0 ; k < 4; k++)
					t2[k] = cos((k+1) * lch.at<Vec3f>(i+1,j)[2]) * lch.at<Vec3f>(i+1,j)[1];
				for(int k = 0; k < 4; k++)
					t2[4 + k] = sin((k+1) * lch.at<Vec3f>(i+1,j)[2]) * lch.at<Vec3f>(i+1,j)[1];
				t2[8] = lch.at<Vec3f>(i+1,j)[1];

				for(int k = 0; k < 9; k++)
					u[k] = t2[k] - t1[k];
			}

			//--------------------------------
			//calculate v
			//--------------------------------
			if(j == 0)
			{
				for(int k = 0 ; k < 4; k++)
					t1[k] = cos((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				for(int k = 0; k < 4; k++)
					t1[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				t1[8] = lch.at<Vec3f>(i,j)[1];

				for(int k = 0 ; k < 4; k++)
					t2[k] = cos((k+1) * lch.at<Vec3f>(i,j+1)[2]) * lch.at<Vec3f>(i,j+1)[1];
				for(int k = 0; k < 4; k++)
					t2[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j+1)[2]) * lch.at<Vec3f>(i,j+1)[1];
				t2[8] = lch.at<Vec3f>(i,j+1)[1];

				for(int k = 0; k < 9; k++)
					v[k] = t2[k] - t1[k];
			}
			else if(j == lch.cols - 1)
			{
				for(int k = 0 ; k < 4; k++)
					t1[k] = cos((k+1) * lch.at<Vec3f>(i,j-1)[2]) * lch.at<Vec3f>(i,j-1)[1];
				for(int k = 0; k < 4; k++)
					t1[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j-1)[2]) * lch.at<Vec3f>(i,j-1)[1];
				t1[8] = lch.at<Vec3f>(i,j-1)[1];

				for(int k = 0 ; k < 4; k++)
					t2[k] = cos((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				for(int k = 0; k < 4; k++)
					t2[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j)[2]) * lch.at<Vec3f>(i,j)[1];
				t2[8] = lch.at<Vec3f>(i,j)[1];

				for(int k = 0; k < 9; k++)
					v[k] = t2[k] - t1[k];
			}
			else
			{
				for(int k = 0 ; k < 4; k++)
					t1[k] = cos((k+1) * lch.at<Vec3f>(i,j-1)[2]) * lch.at<Vec3f>(i,j-1)[1];
				for(int k = 0; k < 4; k++)
					t1[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j-1)[2]) * lch.at<Vec3f>(i,j-1)[1];
				t1[8] = lch.at<Vec3f>(i,j-1)[1];

				for(int k = 0 ; k < 4; k++)
					t2[k] = cos((k+1) * lch.at<Vec3f>(i,j+1)[2]) * lch.at<Vec3f>(i,j+1)[1];
				for(int k = 0; k < 4; k++)
					t2[4 + k] = sin((k+1) * lch.at<Vec3f>(i,j+1)[2]) * lch.at<Vec3f>(i,j+1)[1];
				t2[8] = lch.at<Vec3f>(i,j+1)[1];

				for(int k = 0; k < 9; k++)
					v[k] = t2[k] - t1[k];
			}

			//--------------------------------
			//calculate p , q
			//--------------------------------
			double gx = 0,gy = 0,lx,ly;
			if(i == 0)
			{
				gx = gradient_color(lab.at<Vec3b>(i+1,j), lab.at<Vec3b>(i,j), luv.at<Vec3b>(i+1,j), luv.at<Vec3b>(i,j));
				lx = lch.at<Vec3f>(i+1,j)[0] - lch.at<Vec3f>(i,j)[0];
			}
			else if(i == lab.rows - 1)
			{
				gx = gradient_color(lab.at<Vec3b>(i,j), lab.at<Vec3b>(i-1,j), luv.at<Vec3b>(i,j), luv.at<Vec3b>(i-1,j));
				lx = lch.at<Vec3f>(i,j)[0] - lch.at<Vec3f>(i-1,j)[0];
			}
			else
			{
				gx = gradient_color(lab.at<Vec3b>(i+1,j), lab.at<Vec3b>(i-1,j), luv.at<Vec3b>(i+1,j), luv.at<Vec3b>(i-1,j));
				lx = lch.at<Vec3f>(i+1,j)[0] - lch.at<Vec3f>(i-1,j)[0];
			}
			if(j == 0)
			{
				gy = gradient_color(lab.at<Vec3b>(i,j+1), lab.at<Vec3b>(i,j), luv.at<Vec3b>(i,j+1), luv.at<Vec3b>(i,j));
				ly = lch.at<Vec3f>(i,j+1)[0] - lch.at<Vec3f>(i,j)[0];
			}
			else if( j == lab.cols - 1)
			{
				gy = gradient_color(lab.at<Vec3b>(i,j), lab.at<Vec3b>(i,j-1), luv.at<Vec3b>(i,j), luv.at<Vec3b>(i,j-1));
				ly = lch.at<Vec3f>(i,j)[0] - lch.at<Vec3f>(i,j-1)[0];
			}
			else
			{
				gy = gradient_color(lab.at<Vec3b>(i,j+1), lab.at<Vec3b>(i,j-1), luv.at<Vec3b>(i,j+1), luv.at<Vec3b>(i,j-1));
				ly = lch.at<Vec3f>(i,j+1)[0] - lch.at<Vec3f>(i,j-1)[0];
			}
			p = gx - lx;
			q = gy - ly;

			/*
			//----------------------------------------------------------------------------------------------
			//DEBUG
			//assign value to the mat of gradient for color image and HK_Effect
			//----------------------------------------------------------------------------------------------
			gradientC.at<Vec3b>(i,j)[0] = gradientC.at<Vec3b>(i,j)[1] = gradientC.at<Vec3b>(i,j)[2] = sqrt(gx*gx + gy*gy);
			HK_Effect.at<Vec3b>(i,j)[0] = HK_Effect.at<Vec3b>(i,j)[1] = HK_Effect.at<Vec3b>(i,j)[2] = min(255,max(0,((int)H_KEffect(luv.at<Vec3b>(i,j)))));
			*/

			//--------------------------------
			//calculate Ms
			//--------------------------------
			double tem1[9][9], tem2[9][9];
			for(int m = 0; m < 9; m++)
			{
				for(int n = 0; n < 9; n++)
				{
					tem1[m][n] = u[m] * u[n];
					tem2[m][n] = v[m] * v[n];
				}
			}

			for(int m = 0; m < 9; m++)
			{
				for(int n = 0; n < 9; n++)
				{
					ms[m][n] += (tem1[m][n] + tem2[m][n]);
				}
			}

			//------------------------------
			//calculate bs
			//------------------------------
			for(int k = 0; k < 9; k++)
				bs[k] += (p * u[k] + q * v[k]);
		}
	}

	for(int i = 0; i < 9; i++)
	{
		for(int j = 0; j < 9;j++)
		{
			ms[i][j] += wm[i][j];
		}
	}

	for(int i = 0; i < 9; i++)
		bs[i] += wv[i];

	//----------------------------------
	//calculate X
	//----------------------------------
	MatrixXd m(9,9),b(9,1),b1(9,1);
	for(int i = 0; i < 9; i++)
	{
		b(i) = bs[i];
		b1(i) = bs1[i];
		for(int j = 0; j < 9; j++)
		{
			m(i,j) = ms[i][j];
		}
	}
	m = m + (MatrixXd::Identity(9,9)) * (2 * image.rows * image.cols);
	MatrixXd x = m.inverse();
	x = x * b;

	/*
	//--------------------------------------------------
	//DEBUG
	//print the matrix Ms, Bs and X
	//--------------------------------------------------
	cout<<"-----------------------------------------\nThe value for Ms\n-----------------------------------------\n"<<endl;
	cout<<m<<endl;
	cout<<"-----------------------------------------\nThe value for Bs\n-----------------------------------------\n"<<endl;
	cout<<b<<endl;
	cout<<"-----------------------------------------\nThe value for Xs\n-----------------------------------------\n"<<endl;
	cout<<x<<endl;
	*/

	//----------------------------------
	//for each pixel get the greyscale
	//----------------------------------
	for(int i = 0; i < image.rows; i++)
	{
		for(int j = 0; j < image.cols; j++)
		{
			double result = 0.0;

			//---------------------------------
			//compute the f(h)
			//---------------------------------
			for(int k = 0; k < 4; k++)
			{
				result += cos((k+1) * lch.at<Vec3f>(i,j)[2]) * x(k);
			}
			for(int k = 0; k < 4; k++)
			{
				result += sin((k+1) * lch.at<Vec3f>(i,j)[2]) * x(k + 4);
			}
			result += x(8);

			//----------------------------------------
			//compute L + f(h) * C 
			//----------------------------------------
			result = result * (lch.at<Vec3f>(i,j)[1]);
			result = lch.at<Vec3f>(i,j)[0] + result;
			result = result / 100.0 * 255.0;
			image.at<Vec3b>(i,j)[0] = image.at<Vec3b>(i,j)[1] = image.at<Vec3b>(i,j)[2] = 
        min(255.0,max(0.0,result));
		}
	}
	/*
	//-------------------------------------------------
	//DEBUG
	//Assign value to mat of grayscale gradident
	//------------------------------------------------
	for(int i = 1; i < image.rows - 1; i++)
	{
		for(int j =1; j< image.cols - 1; j++)
		{
			int gx = 0, gy = 0;
			gx = image.at<Vec3b>(i + 1 ,j)[0] - image.at<Vec3b>(i - 1,j)[0];
			gy = image.at<Vec3b>(i ,j + 1)[0] - image.at<Vec3b>(i,j - 1)[0];

			gradientG.at<Vec3b>(i,j)[0] = gradientG.at<Vec3b>(i,j)[1] = gradientG.at<Vec3b>(i,j)[2] = sqrt((double)gx * gx + gy * gy);
		}
	}

	//-------------------------------------------------
	//DEBUG
	//Assign value to mat of gradident difference, already multiply 50
	//------------------------------------------------
	for(int i = 0 ; i< image.rows; i++)
	{
		for(int j = 0; j < image.cols; j++)
		{
			gradient.at<Vec3b>(i,j)[0] = gradient.at<Vec3b>(i,j)[1] = gradient.at<Vec3b>(i,j)[2] = 50 * abs(gradientC.at<Vec3b>(i,j)[0] - gradientG.at<Vec3b>(i,j)[0]);
		}
	}

	//-------------------------------------------------
	//DEBUG
	//show the image for debuging use
	//------------------------------------------------
	imshow("C",gradientC);
	imshow("G",gradientG);
	imshow("MINUS",gradient);
	imshow("HK_Effect", HK_Effect);*/
}

int main(int argc, char* argv[])
{
	string name = "../data/img1.jpg";
	
	Mat image = cv::imread(name);

	if(!image.data)
	{
		cout<<"Can not load image:"<<name<<endl;
		return -1;
	}

	namedWindow("Original Image");
	imshow("Original Image",image);

	Mat result;
	image.copyTo(result);
	clock_t clock_1 = clock();
	color2grayscale(result);
	clock_t clock_2 = clock();
  cout<<(clock_2 - clock_1) / (double)CLOCKS_PER_SEC<<"s"<<endl;

	namedWindow("Output Image");
	imshow("Output Image",result);
	string r_name = "result_" + name + "_random.jpg";
	imwrite(r_name.data(),result);

	waitKey();

	return 0;
}
