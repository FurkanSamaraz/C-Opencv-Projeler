#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Windows.h>
using namespace cv;
using namespace std;

///////////////  Project 3 - License Plate Detector //////////////////////

int main(int argc, char* argv[])
{
	Mat img;
	VideoCapture cap;

	CascadeClassifier plateCascade;
	plateCascade.load("haarcascade_smile.xml");

	if (plateCascade.empty()) { cout << "XML file not loaded" << endl; }

	vector<Rect> plates;
	cap.open(0);
	while (cap.read(img)) {


		plateCascade.detectMultiScale(img, plates, 1.1, 200);//tespit etme

		for (int i = 0; i < plates.size(); i++)
		{
			Mat imgCrop = img(plates[i]);
			//imshow(to_string(i)+"İzinsiz Geçiş", imgCrop);
			//Okunan plakaları dosyaya kayıt etme
			imwrite("Plates/" + to_string(i) + " İzinsiz Plaka " + ".png", imgCrop);
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(0, 0, 255), 3);

			putText(img, "hello word", Point(120, 150), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(0, 0, 255), 3);
		}

		imshow("Gulumseme", img);
		waitKey(1);
	}
	cout << img;

}




//BUTON AMA ÇALIŞMADI
/*
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

Mat on, off;

void fonksiyon(int event, int x ,int y, int flags, void* userdata)
{
	Mat crtim(500, 500, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < off.cols; i++)
	{
		for (int j = 0; j < off.rows; j++)
		{
			crtim.at<Vec3b>(j + 150, i + 150) = off.at<Vec3b>(j, i);
		}
	}

	if (event == EVENT_LBUTTONDOWN)
	{
		if (x>150 && x<150+off.rows && y>150 && y<150+off.cols)
		{
			crtim = 0;
			for (int i = 0; i < on.cols; i++)
			{
				for (int j = 0; j < on.rows; j++)
				{
					crtim.at<Vec3b>(j + 150, i + 150) = on.at<Vec3b>(j, i);
				}
			}
			print("basti");
		}

	}
	imshow("pencere", crtim);
}

int main()
{
	on = imread("on_buton.jpg");
	off = imread("off_buton.jpg");

	namedWindow("pencere", WINDOW_AUTOSIZE);
	setMouseCallback("pencere", fonksiyon, NULL);

	waitKey(0);
	return 0;

}


*/











//Gülümseme tespit
/*
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Windows.h>
using namespace cv;
using namespace std;

///////////////  Project 3 - License Plate Detector //////////////////////

int main(int argc, char* argv[])
{
	Mat img;
	VideoCapture cap;

	CascadeClassifier plateCascade;
	plateCascade.load("haarcascade_smile.xml");

	if (plateCascade.empty()) { cout << "XML file not loaded" << endl; }

	vector<Rect> plates;
	cap.open(0);
	while (cap.read(img)) {


		plateCascade.detectMultiScale(img, plates, 1.1, 200);//tespit etme

		for (int i = 0; i < plates.size(); i++)
		{
			Mat imgCrop = img(plates[i]);
			//imshow(to_string(i)+"İzinsiz Geçiş", imgCrop);
			//Okunan plakaları dosyaya kayıt etme
			imwrite("Plates/" + to_string(i) + " İzinsiz Plaka " + ".png", imgCrop);
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(0, 0, 255), 3);

		}

		imshow("Gulumseme", img);
		waitKey(1);
	}
	cout << img;

}


*/
//İNSAN SAYMA
/*#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
int main(int argc, char* argv[])
{
	CommandLineParser parser(argc, argv, params);

	if (parser.has("help"))
	{
		//yardım bilgilerini yazdır
		parser.printMessage();
	}
	//Arka Plan Çıkarıcı nesneleri oluşturma
	Ptr<BackgroundSubtractor> pBackSub;
	if (parser.get<String>("algo") == "MOG2")
		pBackSub = createBackgroundSubtractorMOG2();
	else
		pBackSub = createBackgroundSubtractorKNN();
	VideoCapture capture(samples::findFile(parser.get<String>("input")));
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open: " << parser.get<String>("input") << endl;
		return 0;
	}
	Mat goruntu, islenmis;
	while (true) {
		capture >> goruntu;
		if (goruntu.empty())
			break;



		//**********************************************************HAREKETLİ OBJE SİYAH EKRAN
		pBackSub->apply(goruntu, islenmis);
		//**********************************************************

		//******************************************************************************************************************NUMARA YAZDIRMA
		rectangle(goruntu, cv::Point(10, 2), cv::Point(100, 20),cv::Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(goruntu, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//******************************************************************************************************************
		

		//mevcut kareyi ve kurbağa maskelerini göster
		imshow("Orijinal", goruntu);
		imshow("FG Mask", islenmis);
		//girişi klavyeden al
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}
	return 0;
}


*/








//BEDEN TESPİT ETME
/*
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void main() {

	Mat img;
	VideoCapture cap(0);

	CascadeClassifier plateCascade;
	plateCascade.load("haarcascade_fullbody.xml");

	if (plateCascade.empty()) { cout << "XML file not loaded" << endl; }

	vector<Rect> plates;

	while (true) {

		cap.read(img);
		plateCascade.detectMultiScale(img, plates, 1.1, 10);

		for (int i = 0; i < plates.size(); i++)
		{
		
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);

		}

		imshow("Image", img);
		waitKey(1);
	}
	cout << img;

}
*/

//Mause log kaydı
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;
#include<iostream>
#include<conio.h>           // Windows kullanmıyorsanız bu satırı değiştirmek veya kaldırmak gerekebilir

// işlev prototipleri ////////////////////////////////////////////////////////////////////////////
cv::Point predictNextPosition(std::vector<cv::Point> &positions);
void mouseMoveCallback(int event, int x, int y, int flags, void* userData);
void drawCross(cv::Mat &img, cv::Point center, cv::Scalar color);

//genel değişkenler ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_RED = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(0.0, 0.0, 255.0);

cv::Point mousePosition(0, 0);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::Mat imgBlank(700, 900, CV_8UC3, cv::Scalar::all(0));            // fareyi üzerinde hareket ettirmek için boş bir görüntü beyan edin

    std::vector<cv::Point> mousePositions;

    cv::Point predictedMousePosition;

    cv::namedWindow("imgBlank");                                // penceresi
    cv::setMouseCallback("imgBlank", mouseMoveCallback);        // 

    while (true) {

        mousePositions.push_back(mousePosition);            // mevcut pozisyonu al

        predictedMousePosition = predictNextPosition(mousePositions);        // sonraki pozisyonu tahmin et

        std::cout << "şu anki pozisyon     = " << mousePositions.back().x << ", " << mousePositions.back().y << "\n";
        std::cout << "sonraki tahmin edilen pozisyon = " << predictedMousePosition.x << ", " << predictedMousePosition.y << "\n";
        std::cout << "--------------------------------------------------\n";

        drawCross(imgBlank, mousePositions.back(), SCALAR_WHITE);
        drawCross(imgBlank, predictedMousePosition, SCALAR_RED);                      // en son tahmin edilen, fiili ve düzeltilmiş pozisyonlarda bir çarpı işareti çizin

        cv::imshow("imgBlank", imgBlank);         // resmi göster

        cv::waitKey(10);                    // işletim sisteminin imgBlank'i yeniden çizmesini sağlamak için bir an duraklayın

        imgBlank = cv::Scalar::all(0);         // Bir dahaki sefere boşluğu boşalt
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Point predictNextPosition(std::vector<cv::Point> &positions) {
    cv::Point predictedPosition;        //bu dönüş değeri olacak
    int numPositions;

    numPositions = positions.size();

    if (numPositions == 0) {

        std::cout << "hata, Sonraki Pozisyonun sıfır puan ile çağrıldığını tahmin et\n";

    } else if (numPositions == 1) {

        return(positions[0]);

    } else if (numPositions == 2) {

        int deltaX = positions[1].x - positions[0].x;
        int deltaY = positions[1].y - positions[0].y;

        predictedPosition.x = positions.back().x + deltaX;
        predictedPosition.y = positions.back().y + deltaY;

    } else if (numPositions == 3) {

        int sumOfXChanges = ((positions[2].x - positions[1].x) * 2) +
            ((positions[1].x - positions[0].x) * 1);

        int deltaX = (int)std::round((float)sumOfXChanges / 3.0);

        int sumOfYChanges = ((positions[2].y - positions[1].y) * 2) +
            ((positions[1].y - positions[0].y) * 1);

        int deltaY = (int)std::round((float)sumOfYChanges / 3.0);

        predictedPosition.x = positions.back().x + deltaX;
        predictedPosition.y = positions.back().y + deltaY;

    } else if (numPositions == 4) {

        int sumOfXChanges = ((positions[3].x - positions[2].x) * 3) +
            ((positions[2].x - positions[1].x) * 2) +
            ((positions[1].x - positions[0].x) * 1);

        int deltaX = (int)std::round((float)sumOfXChanges / 6.0);

        int sumOfYChanges = ((positions[3].y - positions[2].y) * 3) +
            ((positions[2].y - positions[1].y) * 2) +
            ((positions[1].y - positions[0].y) * 1);

        int deltaY = (int)std::round((float)sumOfYChanges / 6.0);

        predictedPosition.x = positions.back().x + deltaX;
        predictedPosition.y = positions.back().y + deltaY;

    } else if (numPositions >= 5) {

        int sumOfXChanges = ((positions[numPositions - 1].x - positions[numPositions - 2].x) * 4) +
            ((positions[numPositions - 2].x - positions[numPositions - 3].x) * 3) +
            ((positions[numPositions - 3].x - positions[numPositions - 4].x) * 2) +
            ((positions[numPositions - 4].x - positions[numPositions - 5].x) * 1);

        int deltaX = (int)std::round((float)sumOfXChanges / 10.0);

        int sumOfYChanges = ((positions[numPositions - 1].y - positions[numPositions - 2].y) * 4) +
            ((positions[numPositions - 2].y - positions[numPositions - 3].y) * 3) +
            ((positions[numPositions - 3].y - positions[numPositions - 4].y) * 2) +
            ((positions[numPositions - 4].y - positions[numPositions - 5].y) * 1);

        int deltaY = (int)std::round((float)sumOfYChanges / 10.0);

        predictedPosition.x = positions.back().x + deltaX;
        predictedPosition.y = positions.back().y + deltaY;

    } else {
        //asla buraya gelmemeli
    }

    return(predictedPosition);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void mouseMoveCallback(int event, int x, int y, int flags, void* userData) {
    if (event == EVENT_LBUTTONDOWN) {
        mousePosition.x = x;
        mousePosition.y = y;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCross(cv::Mat &img, cv::Point center, cv::Scalar color) {
    cv::line(img, cv::Point(center.x - 5, center.y - 5), cv::Point(center.x + 5, center.y + 5), color, 2);
    cv::line(img, cv::Point(center.x + 5, center.y - 5), cv::Point(center.x - 5, center.y + 5), color, 2);

}

*/


//Araç plaka tanıma
/*

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///////////////  Project 3 - License Plate Detector //////////////////////

void main() {

	Mat img;
	VideoCapture cap(0);

	CascadeClassifier plateCascade;
	plateCascade.load("haarcascade_russian_plate_number.xml");

	if (plateCascade.empty()) { cout << "XML file not loaded" << endl; }

	vector<Rect> plates;

	while (true) {

		cap.read(img);
		plateCascade.detectMultiScale(img, plates, 1.1, 10);

		for (int i = 0; i < plates.size(); i++)
		{
			Mat imgCrop = img(plates[i]);
			//imshow(to_string(i)+"İzinsiz Geçiş", imgCrop);
			//Okunan plakaları dosyaya kayıt etme
			imwrite("Plates/" + to_string(i)+" İzinsiz Plaka " + ".png", imgCrop);
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);

		}

		imshow("Image", img);
		waitKey(1);
	}
	cout << img;

}
*/




//Resimden yüz tanıma
/*

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <Windows.h>
using namespace std;
using namespace cv;

int main()
{
	CascadeClassifier yuz;


	vector<Rect> tanima;

	yuz.load("haarcascade_frontalface_default.xml");

	Mat resim = imread("C:/Users/samar/Desktop/image/unnamed.jpg");

	Mat gri;
	cvtColor(resim, gri, COLOR_BGR2GRAY);

	yuz.detectMultiScale(gri, tanima);

	for (int i = 0; i < tanima.size(); i++)
	{
		rectangle(resim, tanima[i], Scalar(255, 0, 0), 3);
	}
	imshow("yüz bulma", resim);
	waitKey(0);
	return 0;

}
*/


/*
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

int main()
{
	//Dahili Webcam için 0 değeri kullanılır
	VideoCapture video(0);
	//Webcam açılmazsa
	if (!video.isOpened())
	{
		cout << "Webcam acilamadi" << endl;
	}
	//Penceremizi oluşturduk
	namedWindow("Webcam Player");

	while (true) //sonsuz dongu
	{
		Mat img;
		//Videoyu frame olarak okuyor
		bool frameOkundu = video.read(img);
		//Okunacak frame kalmadıysa döngüden çıkıyor
		if (!frameOkundu)
		{
			cout << "Okunacak frame kalmadi" << endl;
			break;
		}
		imshow("Webcam Player", img);
		if (waitKey(30) == 27)
		{
			cout << "Esc ile cikis yapildi" << endl;
			destroyWindow("Webcam Player");
			break;
		}
	}
	system("Pause");
	return 0;
}
/*
//Obje okuma
/*
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
int main(int, char**)
{
	VideoCapture cap(0);
	if (!cap.isOpened()) return -1;
	Mat frame, edges;
	namedWindow("edges", WINDOW_AUTOSIZE);
	for (;;)
	{
		cap >> frame;
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);
		if (waitKey(30) >= 0) break;
	}
	return 0;
}
*/


//insan takip etme
/*
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	const string about =
		"This sample demonstrates Lucas-Kanade Optical Flow calculation.\n"
		"The example file can be downloaded from:\n"
		"  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4";
	const string keys =
		"{ h help |      | print this help message }"
		"{ @image | vtest.avi | path to image file }";
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	string filename = samples::findFile(parser.get<string>("@image"));
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	VideoCapture capture(filename);
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	// Create some random colors
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 100; i++)
	{
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r, g, b));
	}
	Mat old_frame, old_gray;
	vector<Point2f> p0, p1;
	// Take first frame and find corners in it
	capture >> old_frame;
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
	goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
	// Create a mask image for drawing purposes
	Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
	while (true) {
		Mat frame, frame_gray;
		capture >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		// calculate optical flow
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
		vector<Point2f> good_new;
		for (uint i = 0; i < p0.size(); i++)
		{
			// Select good points
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				line(mask, p1[i], p0[i], colors[i], 2);
				circle(frame, p1[i], 5, colors[i], -1);
			}
		}
		Mat img;
		add(frame, mask, img);
		imshow("Frame", img);
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
		// Now update the previous frame and previous points
		old_gray = frame_gray.clone();
		p0 = good_new;
	}
}
*/


//insan sayma

/*
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
const char* params
= "{ help h         |           | Print usage }"
"{ input          | vtest.avi | Path to a video or a sequence of image }"
"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
int main(int argc, char* argv[])
{
	CommandLineParser parser(argc, argv, params);
	parser.about("This program shows how to use background subtraction methods provided by "
		" OpenCV. You can process both videos and images.\n");
	if (parser.has("help"))
	{
		//yardım bilgilerini yazdır
		parser.printMessage();
	}
	//Arka Plan Çıkarıcı nesneleri oluşturma
	Ptr<BackgroundSubtractor> pBackSub;
	if (parser.get<String>("algo") == "MOG2")
		pBackSub = createBackgroundSubtractorMOG2();
	else
		pBackSub = createBackgroundSubtractorKNN();
	VideoCapture capture(samples::findFile(parser.get<String>("input")));
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open: " << parser.get<String>("input") << endl;
		return 0;
	}
	Mat goruntu, islenmis;
	while (true) {
		capture >> goruntu;
		if (goruntu.empty())
			break;
		//arka plan modelini güncelle
		pBackSub->apply(goruntu, islenmis);
		//get the frame number and write it on the current frame
		rectangle(goruntu, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(goruntu, frameNumberString.c_str(), cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//mevcut kareyi ve kurbağa maskelerini göster
		imshow("Orijinal", goruntu);
		imshow("FG Mask", islenmis);
		//girişi klavyeden al
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}
	return 0;
}
*/





//YUVARLAK OBJE OKUMA
/*


#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

	VideoCapture cap(0); //videoyu web kamerasından yakala

	if (!cap.isOpened())  // başarılı değilse, programdan çık
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	namedWindow("Control",WND_PROP_AUTOSIZE); //"Kontrol" adlı bir pencere oluştur

	int iLowH = 170;
	int iHighH = 179;

	int iLowS = 150;
	int iHighS = 255;

	int iLowV = 60;
	int iHighV = 255;

	// "Kontrol" penceresinde izleme çubukları oluşturun
	createTrackbar("LowH", "Control", &iLowH, 179); //Ton (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Doyma (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255);//Değer (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	int iLastX = -1;
	int iLastY = -1;

	//Kameradan geçici bir görüntü yakalayın
	Mat imgTmp;
	cap.read(imgTmp);

	//Kamera çıkışı boyutunda siyah bir görüntü oluşturun
	Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);;

	//siyah kare
	while (true)
	{
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal); // videodan yeni bir kare oku



		if (!bSuccess) //başarılı değilse, döngüyü kır
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		Mat imgHSV;

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Yakalanan kareyi BGR'den HSV'ye dönüştürün

		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image


		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//morfolojik kapanma (ön plandaki küçük delikleri kaldırır)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));


		threshold(imgOriginal, imgHSV, 100, 255, cv::THRESH_BINARY_INV);
		//Eşikli görüntünün anlarını hesaplayın
		Moments oMoments = moments(imgThresholded);

		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;
		// alan <= 10000 ise, görüntüde nesne olmadığını ve gürültüden kaynaklandığını düşünürüm, alan sıfır değildir
		if (dArea > 100000)
		{
			//topun konumunu hesapla
			int posX = dM10 / dArea;
			int posY = dM01 / dArea;

			if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
			{
				//Önceki noktadan geçerli noktaya kırmızı bir çizgi çizin ve boyutu
				line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0, 0, 255), 1,8);
			}

			iLastX = posX;
			iLastY = posY;
		}
		//siyah ekran
		imshow("Thresholded Image", imgThresholded); //eşikli resmi göster

		imgOriginal = imgOriginal + imgLines;
		imshow("Original", imgOriginal); //orijinal resmi göster

		if (waitKey(30) == 27) //30ms için 'esc' tuşuna basılmasını bekleyin. 'Esc' tuşuna basılırsa, döngüyü kır
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;
}
*/
















/*
#include <opencv2/opencv.hpp>
using namespace cv;
int main()
{
	VideoCapture video(0);//kamera seçiliyor
	Mat okunan;
	while (true)//sürekli olarak okuma işlemi için gerekli
	{
		video.read(okunan);//frame'ler okunuyor
		imshow("orjinal görüntü", okunan);//okunan değerler gösteriliyor
		if (waitKey(10) == 27)//döngüden çıkma şartı kesinlikle gerekl
		{
			break;
		}
	}
}
*/

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

/*
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;
int main()
{
	cout << "Video Player" << endl;

	string vFileName = "C:/Users/samar/Desktop/Youtube Medya/aaa.mp4";

	string wName = "Video Player";
	cv::VideoCapture vCap;
	vCap.open(vFileName);
	if (vCap.isOpened()) {
		double fps = vCap.get(WINDOW_AUTOSIZE);
		cv::Mat frame;
		cv::namedWindow(wName);
		for (;;)
		{
			if (!vCap.read(frame)) {
				break;
				cv::imshow(wName, frame);
				if (cv::waitKey(1000 / fps) >= 0) {
					break;
				}
				vCap.release();
				cv::destroyWindow(wName);
			}
			else
			{
				cout << "Video file" << vFileName << "Not oponed..." << endl;
			}

		}
	}

}
*/
