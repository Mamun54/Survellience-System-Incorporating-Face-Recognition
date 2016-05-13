/*@author gihan tharanga*/

#include <iostream>
#include <string>
#include <windows.h>
//include opencv core
#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

//file handling
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

static Mat MatNorm(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void dbread(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';'){
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file){
		string error = "no valid input file";
		CV_Error(CV_StsBadArg, error);
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, label);
		if (!path.empty() && !label.empty()){
			images.push_back(imread(path, 0));
			labels.push_back(atoi(label.c_str()));
		}
	}
}

void eigenFaceTrainer(){
	vector<Mat> images;
	vector<int> labels;

	try{
		string filename = "E:/at.txt";
		dbread(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e){
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	//create algorithm eigenface recognizer
	Ptr<FaceRecognizer>  model = createEigenFaceRecognizer();
	//train data
	model->train(images, labels);

	model->save("E:/FDB/yaml/eigenface.yml");

	cout << "Training finished...." << endl;
	////get eigenvalue of eigenface model

	waitKey(10000);
}

void fisherFaceTrainer(){
	/*in this two vector we put the images and labes for training*/
	vector<Mat> images;
	vector<int> labels;

	try{		
		string filename = "csv.ext";
		dbread(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e){
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}


	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

	model->train(images, labels);

	int height = images[0].rows;

	model->save("E:/FDB/yaml/fisherface.yml");

	cout << "Training finished...." << endl;

	Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getMat("eigenvectors");
	// Get the sample mean from the training data
	Mat mean = model->getMat("mean");
	//imshow("mean", MatNorm(mean.reshape(1, images[0].rows)));
	//imwrite(format("%s/mean.png", output_folder.c_str()), MatNorm(mean.reshape(1, images[0].rows)));

	// Display or save the first, at most 16 Fisherfaces:
	
}


//lbpcascades works in lbphrecognier as fast as haarcascades 
int  FaceRecognition(){

	//String name[] = {"Raj","Maruf","Mamun","SIraj","Sir"};
	//int pos_x, pos_y;
	//int v1, v2, v3, v4, v5, v6, v7, v8, v9, v10;

	
	cout << "start recognizing..." << endl;

	//load pre-trained data sets
	Ptr<FaceRecognizer>  model = createFisherFaceRecognizer();
	model->load("E:/FDB/yaml/fisherface.yml");

	Mat testSample = imread("1.jpg", 0);
	

	int img_width = testSample.cols;
	int img_height = testSample.rows;


	//lbpcascades/lbpcascade_frontalface.xml
	string classifier = "C:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
	

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)){
		cout << " Error loading file" << endl;
		return -1;
	}

	VideoCapture cap(0);
	//VideoCapture cap("C:/Users/lsf-admin/Pictures/Camera Roll/video000.mp4");

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	//double fps = cap.get(CV_CAP_PROP_FPS);
	//cout << " Frames per seconds " << fps << endl;
	namedWindow(window, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;
		//cap.read(frame);
		count = count + 1;//count frames;

		if (!frame.empty()){

			//clone from original frame
			original = frame.clone();

			//convert image to gray scale and equalize
			cvtColor(original, graySacleFrame, CV_BGR2GRAY);
			//equalizeHist(graySacleFrame, graySacleFrame);

			//detect face in gray image
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//region of interest
			//cv::Rect roi;

			//person name
			string Pname1 = "";
			string Pname2 = "";
		
			

			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from grya image
				Mat face = graySacleFrame(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				//recognizing what faces detected
				int label = -1; 
				double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << " confidencde " << confidence << endl;

				//drawing green rectagle in recognize face
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				string text = "Get_Face";
			
				
				
				//	bool flag = true;
	//			String get_name1 = " ";
	//			String get_name2 = " ";
				
		//		int flag = 0;
		//		
		//		for (int i = 0; i <5; i++)
		//		{
		//			if ((label == i) && (flag == 0))
		//			{
		//				//string text = format("Person is  = %d", label);
		//				Pname1 = name[i];
		//				get_name1 = Pname1;
		//				flag =1;
		//				label = -1;
		//				break;
  //                     
		//			}
		//			if (flag == 0 && i == 5)
		//			{
		//				Pname1= "Unknown";
		//			}

		//}

		//

		//		
		//		
		//		
		//		
		//	int	flag1 = 0;

		//	//label = -1;
		//		for (int i=4; i>=0; i--)
		//		{
		//			//cout << label << i << endl;

		//			if ((label == i) && (get_name1!=name[i]) && (flag1==0)){
		//				//string text = format("Person is  = %d", label);
		//				Pname2 = name[i];
		//			    // get_name2 = Pname2;
		//				flag1 = 1;
		//				break;
		//			}

		//			if (flag1 ==0)
		//			{
		//				Pname2 = "Unknown";
		//			}
		//		}
		//		  
		//		
		//		
		//	
		//	
				int temp = 0;

				if (label == 2)
				{
					Pname1 = "Mamun";
					//temp = 1;
					//Beep(415, 400);
				}
				else
				{
					Pname1 = "Unknown";
				}
				//if (label == 3)
				//{
					//Pname2 = "Jishan";
					//temp = 1;
					//Beep(415, 400);
				//}
				//else
				//{
				//	Pname2 = "Unknown";
				//}

				
		
			   

			

				int  pos_x = std::max(face_i.tl().x - 10, 0);
				int  pos_y = std::max(face_i.tl().y - 10, 0);

				//name the person who is in the image
				 putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
				 //tText(original,oint(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
				//cv::imwrite("E:/FDB/"+frameset+".jpg", cropImg);

				

			}


			//putText(original, Pname1, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//putText(original, Pname2, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
		

			//putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "Person1: " + Pname1, Point(30, 70), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "Person2: " + Pname2, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//display to the winodw
			cv::imshow(window, original);

			//cout << "model infor " << model->getDouble("threshold") << endl;

		}
		if (waitKey(30) >= 0) break;
	}
}
