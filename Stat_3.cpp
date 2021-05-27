#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

const double infinity = 1000000.;

// Find character index in alphabet
int find_ch(char alphabet[], const int alphabet_size, char ch)
{
	for (int i = 0; i < alphabet_size; ++i)
		if ((alphabet[i] == ch) || ((ch == ' ') && (alphabet[i] == '1')))
			return i;
}

// Or operation
bool or_ab(int a, int b)
{
	return ((a > 0) || (b > 0));
}

// Xor operation
bool xor_ab(int a, int b)
{
	return (((a > 0) && (b == 0)) || ((a == 0) && (b > 0)));
}

int** generate_img(int &width, char alphabet[], const int alphabet_size, string first, string second, const double p, const int ch_h, int* alphabet_width, int*** alphabet_img)
{
	int width_first = 0;
	for (int i = 0; i < first.length(); ++i)
		width_first += alphabet_width[find_ch(alphabet, alphabet_size, first[i])];

	int width_second = 0;
	for (int i = 0; i < second.length(); ++i)
		width_second += alphabet_width[find_ch(alphabet, alphabet_size, second[i])];

	width = max(width_first, width_second);

	int ind_first = 0;
	int cur_width_first = 0;
	int ind_first_ch = find_ch(alphabet, alphabet_size, first[ind_first]);

	int ind_second = 0;
	int cur_width_second = 0;
	int ind_second_ch = find_ch(alphabet, alphabet_size, second[ind_second]);

	int** img = new int* [width];
	for (int x = 0; x < width; ++x)
	{
		img[x] = new int[ch_h];
		for (int y = 0; y < ch_h; ++y)
			img[x][y] = 255 * int(or_ab(alphabet_img[ind_first_ch][cur_width_first][y], alphabet_img[ind_second_ch][cur_width_second][y]));

		cur_width_first++;
		if (cur_width_first == alphabet_width[ind_first_ch])
		{
			cur_width_first = 0;
			ind_first++;
			if (ind_first < first.length())
				ind_first_ch = find_ch(alphabet, alphabet_size, first[ind_first]);
			else
				ind_first_ch = alphabet_size - 1;
		}

		cur_width_second++;
		if (cur_width_second == alphabet_width[ind_second_ch])
		{
			cur_width_second = 0;
			ind_second++;
			if (ind_second < second.length())
				ind_second_ch = find_ch(alphabet, alphabet_size, second[ind_second]);
			else
				ind_second_ch = alphabet_size - 1;
		}
	}

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < ch_h; ++y)
		{
			double r = (double)rand() / RAND_MAX;
			if (r < p)
				img[x][y] = 255 * int(xor_ab(1, img[x][y]));
		}
	}

	return img;
}

// Checks if double is epsilon close to zero (used only for things that are meant to be zeros)
bool is_zero(double a, double epsilon = 0.00001)
{
	return ((a > -epsilon) && (a < epsilon));
}

// Sum logs of probs instead of Mult probs
double probability(const double noise, int pos, const int ch_h, int ch_w, int** alphabet_img, int** img, bool** mask)
{
	double prob = 0.;
	for (int x = pos - ch_w; x < pos; ++x)
		for (int y = 0; y < ch_h; ++y)
			if (mask[x][y])
				prob += (xor_ab(img[x][y], alphabet_img[x - pos + ch_w][y]) ?
					(is_zero(noise) ? -infinity : log(noise)) :
					(is_zero(1. - noise) ? -infinity : log(1. - noise)));
	return prob;
}

// Function which generates letters from the input logs
int generate(int const alphabet_size, double* f, bool show = false)
{
	double* probs = new double[alphabet_size];

	// Separate infinite values from others
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] = ((f[i] < -infinity + 1.) ? 0. : f[i]);

	// Finds the minimal value as a common denominator for all the powers
	double min = infinity;
	for (int i = 0; i < alphabet_size; ++i)
		if (probs[i] < min)
			min = probs[i];
	// For situations, when there is only 1 non-infinity value
	min -= 1;
	// Subtracts that minimal value
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] += (is_zero(probs[i]) ? 0. : -min);

	// Takes exp from powers to convert logs to actual unnormalized probabilities
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] = (is_zero(probs[i]) ? 0. : exp(probs[i]));

	// Summing up all values
	double sum = 0.;
	for (int i = 0; i < alphabet_size; ++i)
		sum += probs[i];

	// Normalizing probabilities
	for (int i = 0; i < alphabet_size; ++i)
		probs[i] /= sum;

	// Generating output
	int result = -1;
	double r = (double)rand() / RAND_MAX;
	double p = 0.;
	while ((p <= r) && (result != alphabet_size - 1))
	{
		result += 1;
		p += probs[result];
	}

	delete[] probs;

	return result;
}

// Sum logs of probs instead of Mult probs
int* solve(int& n,
	const double noise,
	int const alphabet_size,
	const int ch_height,
	const int ch_width,
	double* freq,
	int* alphabet_width,
	const int height,
	const int width,
	int*** alphabet_img,
	int** img,
	bool** mask,
	bool last=false)
{
	// Initialize fs
	double*** fs = new double** [width + 1];
	for (int p = 0; p < width + 1; ++p)
	{
		fs[p] = new double* [alphabet_size]();
		for (int c = 0; c < alphabet_size; ++c)
			fs[p][c] = new double[alphabet_size]();
	}

	// Initialize f
	double** f = new double* [width + 1];
	for (int p = 0; p < width + 1; ++p)
		f[p] = new double[alphabet_size]();

	// f[0][k_0] = p(k_0) = p(k_0 | " ")
	for (int c = 0; c < alphabet_size; ++c)
	{
		//cout << freq[(alphabet_size - 2) * alphabet_size + c] << endl;
		f[0][c] = is_zero(freq[(alphabet_size - 2) * alphabet_size + c]) ? -infinity : log(freq[(alphabet_size - 2) * alphabet_size + c]);
		fs[0][c][alphabet_size - 2] = is_zero(freq[(alphabet_size - 2) * alphabet_size + c]) ? -infinity : log(freq[(alphabet_size - 2) * alphabet_size + c]);
	}

	// Initialize k taken for f[i]
	int** k = new int* [width + 1]();
	for (int p = 0; p < width + 1; ++p)
		k[p] = new int[alphabet_size]();

	// k[0][c] = -1 for safety
	for (int c = 0; c < alphabet_size; ++c)
		k[0][c] = -1;

	// Calculate all f for i in [1, width - 1]
	for (int p = 1; p < width; ++p)
	{
		for (int c = 0; c < alphabet_size; ++c)
		{
			double prob_max = -infinity;
			int k_max = -1;
			// Find max log prob for all possible prev letters
			for (int c_ = 0; c_ < alphabet_size; ++c_)
			{
				int j = p - alphabet_width[c_];
				if (j >= 0)
				{
					double prob_cur = 0.;
					prob_cur += is_zero(freq[c_ * alphabet_size + c]) ? -infinity : log(freq[c_ * alphabet_size + c]);
					prob_cur += probability(noise, p, ch_height, alphabet_width[c_], alphabet_img[c_], img, mask);
					prob_cur += f[j][c_];
					fs[p][c][c_] = prob_cur;
					if (prob_cur > prob_max)
					{
						prob_max = prob_cur;
						k_max = c_;
					}
				}
			}
			// Save max found
			f[p][c] = prob_max;
			k[p][c] = k_max;
		}
	}

	// Calculate f[width][alphabet_size - 2]
	double prob_max = -infinity;
	int k_max = -1;
	// Find max log prob for all possible prev letters
	for (int c_ = 0; c_ < alphabet_size; ++c_)
	{
		int j = width - alphabet_width[c_];
		if (j >= 0)
		{
			double prob_cur = 0.;
			if (last)
			{
				if (c_ != alphabet_size - 1)
					prob_cur += is_zero(freq[c_ * alphabet_size + alphabet_size - 2]) ? -infinity : log(freq[c_ * alphabet_size + alphabet_size - 2]);
			}
			else
			{
				prob_cur += is_zero(freq[c_ * alphabet_size + alphabet_size - 2]) ? -infinity : log(freq[c_ * alphabet_size + alphabet_size - 2]);
			}
			prob_cur += probability(noise, width, ch_height, alphabet_width[c_], alphabet_img[c_], img, mask);
			prob_cur += f[j][c_];
			fs[width][alphabet_size - 2][c_] = prob_cur;
			if (prob_cur > prob_max)
			{
				prob_max = prob_cur;
				k_max = c_;
			}
		}
	}
	// Save max found
	f[width][alphabet_size - 2] = prob_max;
	k[width][alphabet_size - 2] = k_max;
	/*
	for (int p = 0; p < width + 1; ++p)
	{
		cout << p << " ----------------------------------------------- " << endl;
		for (int i = 0; i < alphabet_size; ++i)
		{
			cout << i << " ------------- " << endl;
			for (int j = 0; j < alphabet_size; ++j)
			{
				cout << fs[p][i][j] << " ";
			}
			cout << endl;
		}
	}
	*/
	// Initialize res
	int* res = new int[width + 1]();
	n = 0;

	// Generating answer
	int pos_cur = width;
	int ch_cur = generate(alphabet_size, fs[width][alphabet_size - 2], true);
	while (pos_cur > 0)
	{
		res[n] = ch_cur;
		n++;
		pos_cur -= alphabet_width[ch_cur];
		if (pos_cur > 0)
			ch_cur = generate(alphabet_size, fs[pos_cur][ch_cur]);
	}

	for (int p = 0; p < width + 1; ++p)
		delete[] k[p];
	delete[] k;
	for (int p = 0; p < width + 1; ++p)
		delete[] f[p];
	delete[] f;
	for (int p = 0; p < width + 1; ++p)
	{
		for (int c = 0; c < alphabet_size; ++c)
			delete[] fs[p][c];
		delete[] fs[p];
	}
	delete[] fs;

	return res;
}

bool** generate_mask(int width, int height, char alphabet[], const int alphabet_size, string str, int* alphabet_width, int*** alphabet_img)
{
	bool** img = new bool* [width];
	for (int i = 0; i < width; ++i)
	{
		img[i] = new bool[height];
		for (int j = 0; j < height; ++j)
			img[i][j] = true;
	}

	if (str.length() == 0)
		return img;

	int ind = 0;
	int cur_width = 0;
	int ind_ch = find_ch(alphabet, alphabet_size, str[ind]);
	
	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
			img[x][y] = (alphabet_img[ind_ch][cur_width][y] > 0) ? false : true;

		cur_width++;
		if (cur_width == alphabet_width[ind_ch])
		{
			cur_width = 0;
			ind++;
			if (ind < str.length())
				ind_ch = find_ch(alphabet, alphabet_size, str[ind]);
			else
				ind_ch = alphabet_size - 1;
		}
	}

	return img;
}

// Generates strings
void gibs(string& first,
	string& second,
	int& n1,
	int& n2,
	const int iter,
	const double noise,
	char alphabet[],
	int const alphabet_size,
	const int ch_height,
	const int ch_width,
	double* freq,
	int* alphabet_width,
	const int height,
	const int width,
	int*** alphabet_img,
	int** img,
	bool hack=false,
	bool show=false)
{
	first = "";
	second = "";

	for (int it = 0; it < iter; ++it)
	{
		bool** mask1 = generate_mask(width, height, alphabet, alphabet_size, first, alphabet_width, alphabet_img);
		int* res1 = solve(n2, noise, alphabet_size, ch_height, ch_width, freq, alphabet_width, height, width, alphabet_img, img, mask1, (!hack) || (it == iter - 1));
		/*
		for (int c = n2 - 1; c >= 0; --c)
			cout << ((alphabet[res1[c]] != '1') ? alphabet[res1[c]] : ' ');
		cout << endl;
		*/
		second = "";
		for (int c = n2 - 1; c >= 0; --c)
			if (alphabet[res1[c]] != '2')
				second += ((alphabet[res1[c]] != '1') ? alphabet[res1[c]] : ' ');
			else
				n2--;
		if (show)
		{
			cout << "--- " << it << " ---" << endl;
			cout << second << endl;
		}
		/*
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
				cout << int(mask1[i][j]);
			cout << endl;
		}
		*/
		for (int i = 0; i < width; ++i)
			delete[] mask1[i];
		delete[] mask1;
		delete[] res1;

		bool** mask2 = generate_mask(width, height, alphabet, alphabet_size, second, alphabet_width, alphabet_img);
		int* res2 = solve(n1, noise, alphabet_size, ch_height, ch_width, freq, alphabet_width, height, width, alphabet_img, img, mask2, (!hack) || (it == iter - 1));
		first = "";
		for (int c = n1 - 1; c >= 0; --c)
			if (alphabet[res2[c]] != '2')
				first += ((alphabet[res2[c]] != '1') ? alphabet[res2[c]] : ' ');
			else
				n1--;
		/*
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
				cout << int(mask2[i][j]);
			cout << endl;
		}
		*/
		if (show)
		{
			cout << first << endl;
			cout << "---------" << endl;
		}
		for (int i = 0; i < width; ++i)
			delete[] mask2[i];
		delete[] mask2;
		delete[] res2;
	}
}

int main()
{
	srand(time(NULL));
	const int alphabet_size = 28;
	fstream file;
	file.open("freq.txt", ios::in);

	const int ch_height = 28;
	const int ch_width = 28;

	char alphabet[] = "abcdefghijklmnopqrstuvwxyz12";

	// Freq
	double* freq = new double[alphabet_size * alphabet_size]();
	for (int i = 0; i < alphabet_size - 1; ++i)
		for (int j = 0; j < alphabet_size - 1; ++j)
			file >> freq[i * alphabet_size + j];

	file.close();

	for (int i = 0; i < alphabet_size - 1; ++i)
		freq[i * alphabet_size + (alphabet_size - 1)] = 1.;
	freq[(alphabet_size - 2) * alphabet_size + (alphabet_size - 1)] = 0.;

	for (int j = 0; j < alphabet_size - 1; ++j)
		freq[(alphabet_size - 1) * alphabet_size + j] = 0.;

	freq[(alphabet_size - 1) * alphabet_size + (alphabet_size - 1)] = 1.;

	// Alphabet imgs
	string folder = "alphabet/";
	string suffix = ".png";
	int* alphabet_width = new int[alphabet_size];
	int*** alphabet_img = new int** [alphabet_size];
	for (int c = 0; c < alphabet_size; ++c)
	{
		Mat image;
		string name(1, alphabet[c]);
		image = imread(folder + name + suffix, IMREAD_UNCHANGED);
		alphabet_width[c] = image.size().width;
		alphabet_img[c] = new int* [ch_width];
		for (int x = 0; x < ch_width; ++x)
		{
			alphabet_img[c][x] = new int[ch_height];
			for (int y = 0; y < ch_height; ++y)
			{
				alphabet_img[c][x][y] = int(image.at<uchar>(y, x));
			}
		}
	}

	fstream input;
	input.open("input.txt", ios::in);

	bool hack;
	bool show;
	double noise;
	int iter;
	// Input strings
	string first;
	string second;

	input >> iter;
	cout << "Iterations - " << iter << endl;
	input >> show;
	cout << "Show intermediate results - " << bool(show) << endl;
	input >> hack;
	cout << "Hacks - " << bool(hack) << endl;
	input >> noise;
	cout << "Noise - " << noise << endl;
	getline(input, first);
	getline(input, first);
	cout << "First string - " << first << endl;
	getline(input, second);
	getline(input, second);
	cout << "Second string - " << second << endl;

	input.close();

	// Input img
	int width = 0;
	const int height = ch_height;
	int** img = generate_img(width, alphabet, alphabet_size, first, second, noise, ch_height, alphabet_width, alphabet_img);

	cout << "Generated image" << endl;
	Mat* result = new Mat[3];
	for (int c = 0; c < 3; ++c)
	{
		result[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y)
			{
				result[c].at<uchar>(y, x) = uchar(img[x][y]);
			}
	}
	
	Mat rez;
	vector<Mat> channels;

	channels.push_back(result[0]);
	channels.push_back(result[1]);
	channels.push_back(result[2]);

	merge(channels, rez);

	namedWindow("Generated image", WINDOW_AUTOSIZE);
	imshow("Generated image", rez);
	imwrite("generated_image.png", rez);
	
	int n1 = 0;
	int n2 = 0;
	string str1;
	string str2;
	gibs(str1, str2, n1, n2, iter, noise, alphabet, alphabet_size, ch_height, ch_width, freq, alphabet_width, height, width, alphabet_img, img, hack, show);

	cout << "Results" << endl;
	cout << str1 << endl;
	cout << str2 << endl;

	fstream output;
	output.open("output.txt", ios::out);

	output << str1 << endl;
	output << str2 << endl;

	output.close();

	waitKey(0);
	return 0;
}