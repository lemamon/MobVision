package com.marinho.mobvision;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.BackgroundSubtractorMOG;
import org.opencv.video.BackgroundSubtractorMOG2;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{

    private int                  absoluteSize;
    private File                 mCascadeFile;
    private Mat                  frame;
    private Mat mRgba, mRgbaF, mRgbaT;
    private CascadeClassifier    cascadeClassifier;

    private static final float MAX_WIDTH = 50.0f;
    private static final float MAX_HEIGTH = 50.0f;
    private static final String  TAG = "sim";
    private static final float   CAMERA_SIZE = 2f;
    private Detection detect;
    private List<MatOfPoint> cont;
    private CameraBridgeViewBase mOpenCvCameraView;
   // private BackgroundSubtractorMOG2 bg ;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("sim", "entrou");
        super.onCreate(savedInstanceState);
        //requestWindowFeature(Window.FEATURE_NO_TITLE);
        //getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onPause(){super.onPause();}

    @Override
    public void onResume(){
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_8, this, mLoaderCallback);
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        frame = new Mat();
        detect = new Detection();

       // Toast.makeText(this,""+height, Toast.LENGTH_LONG);
        //frame = new Mat(height, width, CvType.CV_8UC4);
       // mRgbaF = new Mat(height, width, CvType.CV_8UC4);
       // mRgbaT = new Mat(width, width, CvType.CV_8UC4);
       // absoluteSize = (int) (height * CAMERA_SIZE);
    }

    @Override
    public void onCameraViewStopped() { frame.release(); }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        cont = new ArrayList<MatOfPoint>();
        frame = inputFrame.rgba();
        detect.Process(frame);

        cont = detect.getCont();

        Imgproc.drawContours(frame, cont, -1, new Scalar(128,255,255),2);

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        for (int i=0; i<cont.size(); i++)
        {
            MatOfPoint2f contour2f = new MatOfPoint2f( cont.get(i).toArray() );
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

            Rect rect = Imgproc.boundingRect(points);

            if (rect.width > MAX_WIDTH && rect.height> MAX_HEIGTH)
                Core.rectangle(frame, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0, 255, 0), 2);
        }

        //Core.line(frame, new Point(60,120),new Point(120,150),new Scalar(0, 255, 0));
        /*
        Core.transpose(frame, mRgbaT);

        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);

        Core.flip(mRgbaF, frame, 1 );
        */

        return frame;

    }
}
