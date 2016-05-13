/*@author gihan tharanga*/

#include <iostream>
#include <string>
#include <windows.h>
#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

/*header files*/
#include "FaceRec.h"
//#include "VideoCap.h"
//#include "imgProc.h"

using namespace std;
using namespace cv;

int main()
{
	int temp;
	std::cout << "Enter Press: 1 for Training the face"<<endl;
	std::cout << "Enter press: 2 Recogniging the Face"<<endl;
	cout << endl;
	std::cout << "Enter Your Choose:";
	std::cin >> temp;
	
    //fisherFaceTrainer();
	if (temp == 1)
	{
		fisherFaceTrainer();
	}
	if (temp == 2)
	{
		FaceRecognition();
	}
    //int value = FaceRecognition();

	system("pause");
	return 0;
}