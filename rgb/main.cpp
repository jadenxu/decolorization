//-----------------------------------------------------------------
//opencv.cpp is for practice opencv by myself and purpose of urop
//-----------------------------------------------------------------

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<math.h>
#include<Eigen/dense>
using namespace Eigen;
using namespace std;
using namespace cv;

const double PI = 3.14159265;

class point
{
public:
	double l, c, h, u, v, r, g, bl, a, b;
	int num;
	point()
	{
		l = c = h = u = v = r = g = bl = a = b = 0;
		num = 0;
	}
	point(double l, double c, double h, double u, double v, double r, double g, double bl, double a, double b, int n)
	{
		this->l = l;
		this->c = c;
		this->h = h;
		this->u = u;
		this->v = v;
		this->r = r;
		this->g = g;
		this->bl = bl;
		this->a = a;
		this->b = b;
		num = n;
	}
};

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

double gradient_color(Vec3f i, Vec3f j, Vec3f m, Vec3f n, Vec3f p, Vec3f q)
{
	
	double delta_l = (i[0] - j[0]) / 255.0 * 100.0;
	double delta_a = i[1] - j[1];
	double delta_b = i[2] - j[2];

	double delta_r = (p[0] - q[0])/ 254.0 * 100.0;
	double delta_g = (p[1] - q[1])/ 254.0 * 100.0;
	double delta_bl = (p[2] - q[2])/ 254.0 * 100.0;
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
	double result = sqrt(delta_r * delta_r + delta_g * delta_g + delta_bl * delta_bl);
	result = result / sqrt(3.0);
	result = result * sign;

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

	
	//------------------------------------------
	//MY
	//get the 8 average value and the amount, and get the value for w
	//------------------------------------------
	const int NUM_CUB = 8;
	point* rgb = new point[NUM_CUB];
	
	double min_r, max_r, min_g, max_g, min_b, max_b;
	for(int k = 0; k < NUM_CUB; k++)
	{
		if(k&1)
		{
			min_r = 255.0 / 2.0;
			max_r = 256;
		}
		else
		{
			min_r = 0;
			max_r = 255.0 / 2.0;
		}
		
		if(k&2)
		{
			min_g = 255.0 / 2.0;
			max_g = 256;
		}
		else
		{
			min_g = 0;
			max_g = 255.0 / 2.0;
		}

		if(k&4)
		{
			min_b = 255.0 / 2.0;
			max_b = 256;
		}
		else
		{
			min_b = 0;
			max_b = 255.0 / 2.0;
		}

		rgb[k].r = rgb[k].g = rgb[k].bl = rgb[k].l = rgb[k].c = rgb[k].h = rgb[k].u = rgb[k].v = 0.0;
		rgb[k].num = 0;
		for(int i = 0; i < image.rows; i++)
		{
			for(int j = 0; j < image.cols; j++)
			{
				if(image.at<Vec3b>(i,j)[2] >= min_r && image.at<Vec3b>(i,j)[2] < max_r && image.at<Vec3b>(i,j)[1] >= min_g && image.at<Vec3b>(i,j)[1] < max_g && image.at<Vec3b>(i,j)[0] >= min_b && image.at<Vec3b>(i,j)[0] < max_b)
				{
					rgb[k].r += image.at<Vec3b>(i,j)[2];
					rgb[k].g += image.at<Vec3b>(i,j)[1];
					rgb[k].bl += image.at<Vec3b>(i,j)[0];
					rgb[k].l += lab.at<Vec3b>(i,j)[0];
					rgb[k].a += lab.at<Vec3b>(i,j)[1];
					rgb[k].b += lab.at<Vec3b>(i,j)[2];
					rgb[k].c += lch.at<Vec3f>(i,j)[1];
					rgb[k].h += lch.at<Vec3f>(i,j)[2];
					rgb[k].u += luv.at<Vec3b>(i,j)[1];
					rgb[k].v += luv.at<Vec3b>(i,j)[2];
					rgb[k].num++;
				}
			}
		}

		if(rgb[k].num!=0)
		{
			rgb[k].r = rgb[k].r / (double)rgb[k].num;
			rgb[k].g = rgb[k].g / (double)rgb[k].num;
			rgb[k].bl = rgb[k].bl / (double)rgb[k].num;
			rgb[k].l = rgb[k].l / (double)rgb[k].num;
			rgb[k].a = rgb[k].a / (double)rgb[k].num;
			rgb[k].b = rgb[k].b / (double)rgb[k].num;
			rgb[k].c = rgb[k].c / (double)rgb[k].num;
			rgb[k].h = rgb[k].h / (double)rgb[k].num;
			rgb[k].u = rgb[k].u / (double)rgb[k].num;
			rgb[k].v = rgb[k].v / (double)rgb[k].num;
		}
	}

	//for(int i = 0; i < NUM_CUB; i++)
		//cout<<rgb[i].bl <<" "<<rgb[i].g<<" "<<rgb[i].r <<" "<<rgb[i].num<<endl<<endl;//<<" "<<rgb[i].l<<" "<<rgb[i].c <<" "<<rgb[i].h<<" "<<rgb[i].u <<" "<<rgb[i].v<<" "<<rgb[i].a <<" "<<rgb[i].b<<" "<<rgb[i].num<<endl;
	double w[9], tem1[9], tem2[9],wm[9][9],wv[9];

	for(int i = 0; i < 9; i++)
	{
		w[i] = wv[i] = 0;
		for(int j = 0; j < 9; j++)
		{
			wm[i][j] = 0;
		}
	}

	for(int i = 0; i < NUM_CUB; i++)
	{
		for(int j = 0; j < NUM_CUB; j++)
		{
			if(j > i)
			{
				for(int k = 0 ; k < 4; k++)
					tem1[k] = cos((k+1) * rgb[i].h) * rgb[i].c;
				for(int k = 0; k < 4; k++)
					tem1[4 + k] = sin((k+1) * rgb[i].h) * rgb[i].c;
				tem1[8] = rgb[i].c;

				for(int k = 0 ; k < 4; k++)
					tem2[k] = cos((k+1) * rgb[j].h) * rgb[j].c;
				for(int k = 0; k < 4; k++)
					tem2[4 + k] = sin((k+1) * rgb[j].h) * rgb[j].c;
				tem2[8] = rgb[j].c;

				for(int k = 0; k < 9 ;k++)
					w[k] = tem1[k] - tem2[k];
				
				//double weight = (double)min(rgb[i].num , rgb[j].num) / ((NUM_CUB - 1) * 0.5);
				double weight = ((double)rgb[i].num * rgb[j].num) / (0.5 * luv.cols * luv.rows);

				for(int m = 0; m < 9; m++)
				{
					for(int n = 0; n < 9; n++)
					{
						wm[m][n] += weight * w[m] * w[n];
					}
				}

				Vec3f rgb_i(rgb[i].r, rgb[i].g, rgb[i].bl);
				Vec3f rgb_j(rgb[j].r, rgb[j].g, rgb[j].bl);
				Vec3f lab_i(rgb[i].l, rgb[i].a, rgb[i].b);
				Vec3f lab_j(rgb[j].l, rgb[j].a, rgb[j].b);
				Vec3f luv_i(rgb[i].l, rgb[i].u, rgb[i].v);
				Vec3f luv_j(rgb[j].l, rgb[j].u, rgb[j].v);
				double r = gradient_color(lab_i,lab_j,luv_i,luv_j,rgb_i, rgb_j) - (rgb[i].l - rgb[j].l)/ 255.0 * 100.0;

				for(int k = 0; k < 9; k++)
				{
					wv[k] += weight * r * w[k];
				}
			}
		}
	}


	/*cout<<"-----------------------------------------\nThe value for W\n-----------------------------------------\n"<<endl;
	for(int i = 0; i < 9; i++)
		cout<<wv[i]<<endl;
	cout<<endl;*/
	

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
				gx = gradient_color(lab.at<Vec3b>(i+1,j), lab.at<Vec3b>(i,j), luv.at<Vec3b>(i+1,j), luv.at<Vec3b>(i,j),image.at<Vec3b>(i+1,j), image.at<Vec3b>(i,j));
				lx = lch.at<Vec3f>(i+1,j)[0] - lch.at<Vec3f>(i,j)[0];
			}
			else if(i == lab.rows - 1)
			{
				gx = gradient_color(lab.at<Vec3b>(i,j), lab.at<Vec3b>(i-1,j), luv.at<Vec3b>(i,j), luv.at<Vec3b>(i-1,j),image.at<Vec3b>(i,j),image.at<Vec3b>(i-1,j));
				lx = lch.at<Vec3f>(i,j)[0] - lch.at<Vec3f>(i-1,j)[0];
			}
			else
			{
				gx = gradient_color(lab.at<Vec3b>(i+1,j), lab.at<Vec3b>(i-1,j), luv.at<Vec3b>(i+1,j), luv.at<Vec3b>(i-1,j),image.at<Vec3b>(i+1,j),image.at<Vec3b>(i-1,j));
				lx = lch.at<Vec3f>(i+1,j)[0] - lch.at<Vec3f>(i-1,j)[0];
			}
			if(j == 0)
			{
				gy = gradient_color(lab.at<Vec3b>(i,j+1), lab.at<Vec3b>(i,j), luv.at<Vec3b>(i,j+1), luv.at<Vec3b>(i,j),image.at<Vec3b>(i,j+1), image.at<Vec3b>(i,j));
				ly = lch.at<Vec3f>(i,j+1)[0] - lch.at<Vec3f>(i,j)[0];
			}
			else if( j == lab.cols - 1)
			{
				gy = gradient_color(lab.at<Vec3b>(i,j), lab.at<Vec3b>(i,j-1), luv.at<Vec3b>(i,j), luv.at<Vec3b>(i,j-1),image.at<Vec3b>(i,j),image.at<Vec3b>(i,j-1));
				ly = lch.at<Vec3f>(i,j)[0] - lch.at<Vec3f>(i,j-1)[0];
			}
			else
			{
				gy = gradient_color(lab.at<Vec3b>(i,j+1), lab.at<Vec3b>(i,j-1), luv.at<Vec3b>(i,j+1), luv.at<Vec3b>(i,j-1),image.at<Vec3b>(i,j+1),image.at<Vec3b>(i,j-1));
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

	for(int k = 0; k < 9; k++)
		bs[k] += wv[k];

	for(int m = 0; m < 9; m++)
	{
		for(int n = 0; n < 9; n++)
		{
			ms[m][n] += wm[m][n];
		}
	}

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
	string r_name = "result_" + name + "_rgb1.jpg";
	imwrite(r_name.data(),result);

	waitKey();

	return 0;
}
