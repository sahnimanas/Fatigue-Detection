#include<vector>
#include<Windows.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#define CHECK_PERIOD 10


using namespace dlib;
using namespace std;
using namespace cv;

double computeavg(std::vector<double> &window)
{
	double avg = 0.0;
	for (std::vector<double>::iterator i = window.begin(); i != window.end(); i++)
	{
		avg += *i;
	}

	avg /= window.size();
	return avg;
}

void disp(std::vector<double> &window)
{
	cout << computeavg(window) << "-------";
	for (std::vector<double>::iterator i = window.begin(); i != window.end(); i++)
		cout << *i << "\t";
	cout << endl;
}

void calibrate(std::vector<double> &pastRatios, VideoCapture &cap, frontal_face_detector &detector, shape_predictor &pose_model, image_window &win)
{
	cout << "Collecting calibration info\n";
	Mat temp;
	double ratio = 0;
	for (int i = 0; i < CHECK_PERIOD * 3; i++)
	{
		cap >> temp;
		cv_image<bgr_pixel> cimg(temp);

		// Detect faces 
		std::vector<rectangle> faces = detector(cimg);

		// Find the pose of each face.
		std::vector<full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));

		win.clear_overlay();
		if (shapes.empty())
		{
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
			i--;
			continue;
		}

		for (int i = 36; i <= 47; i++)
			draw_solid_circle(cimg, shapes[0].part(i), 2.0, bgr_pixel(0, 0, 255));
		win.set_image(cimg);
		win.add_overlay(render_face_detections(shapes));
		
		if (i < CHECK_PERIOD)
			continue;
		
		ratio = (shapes[0].part(36) - shapes[0].part(39)).length_squared();
		ratio /= (shapes[0].part(37) - shapes[0].part(41)).length_squared();
		
		pastRatios.push_back(ratio);
		
		disp(pastRatios);
	}
	cout << "------------------DONE CALIBRATING------------------\n";
}

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("C:/dlib-19.2/python_examples/shape_predictor_68_face_landmarks.dat") >> pose_model;
		double ratio;
		Mat temp;
        // Grab and process frames until the main window is closed by the user.
		std::vector<double> window;
		calibrate(window, cap, detector, pose_model, win);
		double runningAvg = computeavg(window);
        while(!win.is_closed())
        {
            // Grab a frame
            cap >> temp;
            cv_image<bgr_pixel> cimg(temp);
			
            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);

			// Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));
			
			win.clear_overlay();
			if (shapes.empty())
			{
				win.set_image(cimg);
				win.add_overlay(render_face_detections(shapes));
				continue;
			}
			
			for (int i = 36; i <= 47; i++)
				draw_solid_circle(cimg, shapes[0].part(i), 2.0, bgr_pixel(0, 0, 255));
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
			
			ratio = (shapes[0].part(36) - shapes[0].part(39)).length_squared();
			ratio /= (shapes[0].part(37) - shapes[0].part(41)).length_squared();

			runningAvg += (ratio - window.front()) / CHECK_PERIOD;
			window.push_back(ratio);
			window.erase(window.begin());
			
			disp(window);
        }
    }
    catch(serialization_error& e)
    {
        cout << "Need landmarks.dat file first" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
