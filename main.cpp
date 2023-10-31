#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <unistd.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "multilayer_stixel_world.h"
#include "semi_global_matching.h"

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v * (1.f - s);
	tab[2] = v * (1.f - s * h);
	tab[3] = v * (1.f - s * (1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp = 64, float offset = 0)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp + offset, maxdisp) / maxdisp);
}

void writeStixelsToCSV(const std::string &filename, const std::string &output_path, const std::vector<Stixel> &stixels) {
    std::ofstream file(output_path + filename + ".csv");
    std::string image_path = "testing/STEREO_LEFT/";
    float baseline = 0.2122111542368276;
    float focal = 6.0;
    if (!file.is_open()) {
        std::cerr << "Fail to load .csv file for stixel: " << filename << std::endl;
        return;
    }
    // header
    file << "img_path,x,y,class,depth,y_top\n";
    std::string class_num = "0";

    // write stixel by stixel u means col, v means row
    for (const auto &stixel : stixels) {
        int x = (stixel.u / 8) * 8;
        int y = (stixel.vB / 8) * 8;
        float dist = 0.0;
        if (stixel.disp != 0.0) {
            dist = baseline * focal / stixel.disp;
        }
        file << image_path << filename << ".png" << ","
             << x << ","         // col
             << y << ","        // row-bottom
             << class_num << ","     // class
             << dist << ","     // disparity
             << stixel.vT << "\n";       // row-top
    }
    file.close();
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
	cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
}

class SGMWrapper
{

public:

	SGMWrapper(int numDisparities)
	{
		SemiGlobalMatching::Parameters param;
		param.numDisparities = numDisparities / 2;
		param.max12Diff = -1;
		param.medianKernelSize = -1;
		sgm_ = cv::Ptr<SemiGlobalMatching>(new SemiGlobalMatching(param));
	}

	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1)
	{
		cv::pyrDown(I1, I1_);
		cv::pyrDown(I2, I2_);

		sgm_->compute(I1_, I2_, D1_, D2_);

		cv::resize(D1_, D1, I1.size(), 0, 0, cv::INTER_CUBIC);
		cv::resize(D2_, D2, I1.size(), 0, 0, cv::INTER_CUBIC);
		D1 *= 2;
		D2 *= 2;
		cv::medianBlur(D1, D1, 3);
		cv::medianBlur(D2, D2, 3);
		SemiGlobalMatching::LRConsistencyCheck(D1, D2, 5);
	}

private:
	cv::Mat I1_, I2_, D1_, D2_, D2;
	cv::Ptr<SemiGlobalMatching> sgm_;
};

int main(int argc, char* argv[]) {
    if (chdir("../") != 0) {
        perror("chdir");
        return -1;
    }
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working dir: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
        return -1;
    }

    // std::string config_file = "config.yaml";
    // cv::FileStorage config(config_file, cv::FileStorage::READ);
    // if (!config.isOpened()) {
    //     std::cerr << "Unable to open " << config_file << std::endl;
    //     return -1;
    // }
    std::string left_img_path, right_img_path, calib_file, output_path, phase;

    left_img_path = "/home/marcel/datasets/ameise_okt23/validation/STEREO_LEFT/";
    right_img_path = "/home/marcel/datasets/ameise_okt23/validation/STEREO_RIGHT/";
    calib_file = "ameise.xml";
    output_path = "/home/marcel/datasets/ameise_okt23/validation/targets_from_stereo/";
    phase = "testing";
    //config.release();

	// stereo SGBM
	const int numDisparities = 128;
	SGMWrapper sgm(numDisparities);

	// read camera parameters
    std::string calib_base_path = "calibration/";
	const cv::FileStorage fs(calib_base_path + calib_file, cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	// input parameters
	MultiLayerStixelWorld::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.camera.height = fs["Height"];
	param.camera.tilt = fs["Tilt"];
	param.dmax = numDisparities;

	cv::Mat disparity;
	MultiLayerStixelWorld stixelWorld(param);

    std::vector<std::string> pngFiles;
    for (const auto &entry : std::filesystem::directory_iterator(left_img_path)) {
        if (entry.path().extension() == ".png") {
            pngFiles.push_back(entry.path().stem().string());
        }
    }
    std::cout << pngFiles.size() << " files found!" << std::endl;

	for (const auto& filename : pngFiles) {
		// e.g. left/frame_num_%09d.pgm
        // frame_81_id00101-id00151_1696953358541433071-1696953363441302825-22-1.png
		cv::Mat I1 = cv::imread(left_img_path + filename + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat I2 = cv::imread(right_img_path + filename + ".png", cv::IMREAD_GRAYSCALE);

		if (I1.empty() || I2.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
		CV_Assert(I1.type() == CV_8U || I1.type() == CV_16U);

		if (I1.type() == CV_16U)
		{
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
		}

		const auto t1 = std::chrono::steady_clock::now();

		// compute dispaliry
		sgm.compute(I1, I2, disparity);
		disparity.convertTo(disparity, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);

		// compute stixels
		const auto t2 = std::chrono::steady_clock::now();

		std::vector<Stixel> stixels;
		stixelWorld.compute(disparity, stixels);

		const auto t3 = std::chrono::steady_clock::now();
		const auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

		// colorize disparity
		cv::Mat disparityColor;
		disparity.convertTo(disparityColor, CV_8U, 255. / numDisparities);
		cv::applyColorMap(disparityColor, disparityColor, cv::COLORMAP_JET);
		disparityColor.setTo(cv::Scalar::all(0), disparity < 0);

		// put processing time
		cv::putText(disparityColor, cv::format("dispaliry computation time: %4.1f [msec]", 1e-3 * duration12),
			cv::Point(100, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(disparityColor, cv::format("stixel computation time: %4.1f [msec]", 1e-3 * duration23),
			cv::Point(100, 80), 2, 0.75, cv::Scalar(255, 255, 255));

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(I1, draw, cv::COLOR_GRAY2BGR);

		cv::Mat stixelImg = cv::Mat::zeros(I1.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp));
		cv::addWeighted(draw, 1, stixelImg, 0.5, 0, draw);
		// write to csv
		writeStixelsToCSV(filename, output_path, stixels);
		// highGUI with opencv
		// cv::imshow("disparity", disparityColor);
		// cv::imshow("stixels", draw);
		//cv::imwrite("disparity.png", disparityColor);
		//cv::imwrite(output_path + "proof/" + filename + "_stixel.png", draw);
		std::cout << filename << " finished." << std::endl;
		
		// const char c = cv::waitKey(1);
		// if (c == 27)
		// 	break;
		// if (c == 'p')
		// 	cv::waitKey(0);
	}
    return 0;
}
