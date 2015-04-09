package com.marinho.mobvision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by 002820 on 08/04/2015.
 */
public class Detection {
    //private Mat frame;
    private List<MatOfPoint> cont;
    private BackgroundSubtractorMOG2 bg;
    private Mat firstPlan;

    public void Process(Mat inputFrame){
        cont        = new ArrayList<MatOfPoint>();
        bg          = new BackgroundSubtractorMOG2();
        firstPlan   = new Mat();
        Mat dist = new Mat();
        Mat fg = new Mat();
        Mat bgr = new Mat();

        bg.apply(inputFrame, firstPlan,0.01);

        //Imgproc.morphologyEx(firstPlan, Imgproc.MORPH_OPEN, new Mat(), 2);
        //Imgproc.erode(firstPlan, firstPlan, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
        //Imgproc.dilate(firstPlan, firstPlan, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));

        Imgproc.erode(firstPlan, firstPlan, Imgproc.getStructuringElement(Imgproc.MORPH_OPEN, new Size(2,2)));
        Imgproc.dilate(firstPlan, bgr, Imgproc.getStructuringElement(Imgproc.MORPH_OPEN, new Size(2, 2)));

        Imgproc.distanceTransform(firstPlan, dist, Imgproc.DIST_LABEL_PIXEL,3 );
        Imgproc.findContours(firstPlan, cont, new Mat(),Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        //Imgproc.threshold(dist, fg, 0.7 * dist.max);

    }

    public List<MatOfPoint> getCont(){return cont;}
}
