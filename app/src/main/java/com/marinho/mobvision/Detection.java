package com.marinho.mobvision;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
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
        Mat mask = new Mat();



        Imgproc.cvtColor(inputFrame, mask, Imgproc.COLOR_RGBA2RGB);
        bg.apply(mask, mask,0.05);

        Mat kernel = Mat.ones(new Size(3,3), CvType.CV_8U);
        Imgproc.morphologyEx(mask,mask,Imgproc.MORPH_OPEN,kernel,new Point(),2);

        Mat bgr =new Mat();
        Imgproc.dilate(mask, bgr,kernel,new Point(),3);

        Mat dist = new Mat();
        Imgproc.distanceTransform(mask, dist, Imgproc.DIST_LABEL_PIXEL, 3);

        Mat fg = new Mat();
        Imgproc.threshold(dist,fg,0.7 * dist.elemSize(),255,0);

        Core.subtract(bgr,fg,mask);

        Imgproc.findContours(mask, cont, new Mat(),Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);


        //Imgproc.cvtColor(inputFrame, firstPlan, Imgproc.COLOR_RGBA2RGB); //the apply function will throw the above error if you don't feed it an RGB image
        //bg.apply(firstPlan, mask, 0.05); //apply() exports a gray image by definition
        //Imgproc.cvtColor(mask, inputFrame, Imgproc.COLOR_GRAY2RGBA);


        //Imgproc.morphologyEx(firstPlan, Imgproc.MORPH_OPEN, new Mat(), 2);
        //Imgproc.erode(firstPlan, firstPlan, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
        //Imgproc.dilate(firstPlan, firstPlan, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));

      //  Imgproc.erode(firstPlan, firstPlan, Imgproc.getStructuringElement(Imgproc.MORPH_OPEN, new Size(2,2)));
      //  Imgproc.dilate(firstPlan, bgr, Imgproc.getStructuringElement(Imgproc.MORPH_OPEN, new Size(2, 2)));

      //  Imgproc.distanceTransform(firstPlan, dist, Imgproc.DIST_LABEL_PIXEL,3 );

        //Imgproc.threshold(dist, fg, 0.7 * dist.max);

        //return mask;
    }

    public List<MatOfPoint> getCont(){return cont;}
}
