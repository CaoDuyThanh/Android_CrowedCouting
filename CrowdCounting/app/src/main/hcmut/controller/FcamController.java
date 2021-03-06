package hcmut.controller;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.media.AudioManager;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Environment;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import hcmut.UI.Cam;
import hcmut.UI.CustomDialog;
import hcmut.aclab.crowd.counting.R;
import hcmut.activity.CountingMain;
import hcmut.data.Const;
import hcmut.framework.BasicFlow;
import hcmut.framework.data.RequestFW;
import hcmut.framework.data.ResponseFW;
import hcmut.framework.lib.AppLibFile;
import hcmut.framework.lib.AppLibGeneral;

/**
 * Created by minh on 09/08/2015.
 */
public class FcamController extends BasicFlow {
    private CountingMain fcam;

    static{ System.loadLibrary("opencv_java3"); }

    public FcamController(CountingMain fcam) {
        super(fcam, BasicFlow.CONTROLLER);
        this.fcam = fcam;
        initController();
    }

    private void initController() {
    }

    @Override
    public void listenToRequest(RequestFW request) {
        switch (request.getCode()) {
            case Const.REQ_TAKE_PICTURE_NOW:
                String savePath = AppLibFile.getExternalStoragePath(Const.PICTURE_FOLDER);
                if(AppLibGeneral.isEmptyString(savePath)) {
                    String picture_path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath();
                    picture_path += picture_path.endsWith(File.separator)?"":File.separator;
                    picture_path += fcam.getString(R.string.app_name);
                    AppLibFile.createDirIfNotExists(picture_path);
                    savePath = picture_path;
                }
                File pictureFileDir = new File(savePath);
                if (!pictureFileDir.exists() && !pictureFileDir.mkdirs()) {
                    Toast.makeText(fcam, "Cannot create directory to save picture!",
                            Toast.LENGTH_LONG).show();
                    break;
                }
                takePicture(savePath);
                break;
            case Const.REQ_ESTIMATION:
                // forward the request to virtual server
                String feature = extractFeature(fcam.current_bitmap);
                fcam.current_feature = feature;
                request.setData(feature);
                Toast.makeText(fcam, "Feature vector = " + fcam.current_feature, Toast.LENGTH_LONG).show();
                sendRequest(request);
                break;

            case Const.REQ_GALLERY:
                Intent galleryIntent = new Intent(
                        Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                fcam.releaseCamera();
                fcam.startActivityForResult(galleryIntent, Const.ACTIVITY_RESULT_GALLERY_CODE);
                break;

            default:
                break;
        }
    }

    @Override
    public void listenToResponse(ResponseFW response) {
        switch (response.getCode()) {
            case Const.RESP_ESTIMATION:
                sendResponse(response);
                break;
            default:
                break;
        }
    }

    private String double2string(double[] vector) {
        String result = "";
        for(int i=0; i<vector.length; i++) {
            result += result.length()==0?String.valueOf(vector[i]):" "+String.valueOf(vector[i]);
        }
        return result;
    }

    private String toJSON(double[] vector) throws JSONException{
        ArrayList<Double> arrayList = new ArrayList<Double>();
        for (double value: vector) {
            arrayList.add(value);
        }

        JSONObject jsonObject = new JSONObject();
        JSONArray jsonArray = new JSONArray(arrayList);
        jsonObject.put("features", jsonArray);
        return jsonObject.toString();
    }

    public String extractFeature(Bitmap input) {
        // resize to desired size
        int kernel_size = Const.KERNEL_SIZE;
        int width = Const.IMG_RESIZE_WIDTH;
        int height = Const.IMG_RESIZE_HEIGHT;
        Bitmap resized = Bitmap.createScaledBitmap(input, width, height, true);
        // convert to Mat (OpenCV datatype)
        Mat img = new Mat();
        Utils.bitmapToMat(resized, img);
        // convert to grayscale
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        // histogram equalization
        //Imgproc.equalizeHist(img, img);

        // scale img to double 0->1

        img.convertTo(img, CvType.CV_32F);
        Core.MinMaxLocResult minMax = Core.minMaxLoc(img);
        Core.subtract(img, new MatOfDouble(minMax.minVal), img);
        Core.multiply(img, new MatOfDouble(1f/(minMax.maxVal - minMax.minVal)), img);


        // generate Gabor kernel
        // params
        double sigma;  // tmp value
        double[] theta = new double[] {0, Math.PI/4, Math.PI/2, 3*Math.PI/4};
        double[] lambd = new double[] {0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f};
        double gamma = 1.0f;
        double psi = 0.0f;

        double[] vector = new double[2*theta.length*lambd.length];

        int count = 0;
        for(int t = 0; t < theta.length; t++) {
            for(int l = 0; l < lambd.length; l++) {
                sigma = lambd[l] * kernel_size/2;
                Mat gaborKernel = Imgproc.getGaborKernel(new Size(kernel_size, kernel_size),
                        sigma, theta[t], lambd[l], gamma, psi, CvType.CV_32F);
                        
                Core.multiply(gaborKernel, new MatOfDouble(1f/(2*Math.PI*(sigma*sigma))), gaborKernel);  
                        
                Mat imgAtKernel = img.clone();
                imgAtKernel.convertTo(imgAtKernel, CvType.CV_32F);
                Imgproc.filter2D(imgAtKernel, imgAtKernel, CvType.CV_32F, gaborKernel);
                // calculate mean and variance
                MatOfDouble mean1 = new MatOfDouble();
                MatOfDouble std1 = new MatOfDouble();
                Core.meanStdDev(imgAtKernel, mean1, std1);
                double[] temp1 = mean1.get(0, 0);
                double mean = temp1[0];
                double[] temp2 = std1.get(0, 0);
                double std = temp2[0];
                double variance = std * std;

                vector[count] = AppLibGeneral.round(mean, Const.ROUND_DIGITS);
                vector[count+1] = AppLibGeneral.round(variance, Const.ROUND_DIGITS);
                count += 2;
            }
        }

        /*double sig = 15.5, th = 0, lm = 0.5, gm = 1, ps = 0;
        Mat kernel = Imgproc.getGaborKernel(new Size(kernel_size, kernel_size), sig, th, lm, gm, ps, CvType.CV_32F);
        //Mat resultMat = new Mat();
        img.convertTo(img, CvType.CV_32F);
        Imgproc.filter2D(img, img, CvType.CV_32F, kernel);*/
        
        // calculate mean and variance


        //Core.MinMaxLocResult minMaxKernel = Core.minMaxLoc(img);

        //Toast.makeText(fcam, "Feature vector = " + fcam.current_feature, Toast.LENGTH_LONG).show();

        /*
        // visualize the image / filtered image
        Core.MinMaxLocResult minMax = Core.minMaxLoc(img);
        Core.subtract(img, new MatOfDouble(minMax.minVal), img);
        Core.multiply(img, new MatOfDouble(255f/(minMax.maxVal - minMax.minVal)), img);
        img.convertTo(img, CvType.CV_8UC1);
        */

        //Imgproc.cvtColor(imgToProcess, imgToProcess, Imgproc.COLOR_BGR2GRAY);
        //Imgproc.cvtColor(imgToProcess, imgToProcess, Imgproc.COLOR_GRAY2RGBA, 4);

        /*
        Mat dstMat = new Mat(Const.IMG_RESIZE_WIDTH, Const.IMG_RESIZE_HEIGHT, CvType.CV_8UC1);
        //Mat gaborKernel = new Mat(24, 24, CvType.CV_8UC1);
        //Imgproc.filter2D(srcMat, dstMat, -1, gaborKernel);

        Mat matA = new Mat(3,3,CvType.CV_32F,new Scalar(1));
        Mat matB = new Mat(3,3,CvType.CV_32F,new Scalar(25));
        Mat ker = new Mat(3,3,CvType.CV_32F);
        Core.divide(matA, matB, ker);
        Imgproc.filter2D(srcMat, dstMat, -1, ker);
        */

        /*Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, result);
        return result;*/

        String featureJsonObject = "";
        try {
            featureJsonObject = toJSON(vector);
        }catch (JSONException e){
            // Catch the exception and handle it
        }

        return featureJsonObject;

        //return double2string(vector);
    }


    public void takePicture(final String savePath) {

        final Camera cam = fcam.getCamera();

        final Context context = fcam.getApplicationContext();
        // get the app settings
        final boolean isSilent = AppLibGeneral.getConfigurationBoolean(fcam, Const.PREF_SETTINGS, Const.SETTINGS_SILENT_MODE, true);
        final AudioManager mgr = (AudioManager) fcam.getSystemService(Context.AUDIO_SERVICE);
        if(isSilent && mgr!=null) {  // turn off the sound
            //mgr.setStreamMute(AudioManager.STREAM_SYSTEM, true);
            Cam.disableShutterCompletely(mgr, cam);
        }
        cam.takePicture(null, new Camera.PictureCallback() {
            @Override
            public void onPictureTaken(byte[] data, Camera camera) {
                // update floating UI (final tick)
                //fcam.startService(new Intent(context, FcamService.class)
                //        .setFlags(FcamService.FLAG_TAKE_PICTURE_DONE));
            }
        }, new Camera.PictureCallback() {
            @Override
            public void onPictureTaken(byte[] data, Camera camera) {
                if (data != null) {
                    /*Bitmap b = BitmapFactory.decodeByteArray(data, 0, data.length);
                    b = AppLibFile.ccw(b, Cam.getCameraRotatingAngle(fcam, Camera.CameraInfo.CAMERA_FACING_BACK, camera));
                    String imgPath = AppLibFile.writeBitmapToExternalStorage(fcam, b, Const.PICTURE_QUALITY, savePath);*/

                    Bitmap b = BitmapFactory.decodeByteArray(data, 0, data.length);
                    fcam.current_bitmap = b;
                    //** extract feature and write to storage
                    //Bitmap bitmapResult = extractFeature(b);
                    //AppLibFile.writeBitmapToExternalStorage(fcam, bitmapResult, Const.PICTURE_QUALITY, savePath);


                    String timeNow = AppLibGeneral.getTimeNow();
                    String pictureName = timeNow + ".jpg";
                    String folderPath = savePath;
                    folderPath += (savePath.endsWith(File.separator))?"":File.separator;
                    final String imgPath = folderPath + pictureName;

                    try {
                        FileOutputStream fos = new FileOutputStream(new File(imgPath));
                        fos.write(data);
                        fos.close();
                        // update user gallery
                        File imgFile = new File(imgPath);
                        fcam.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, Uri.fromFile(imgFile)));
                    } catch (Exception error) {
                        Toast.makeText(context, "Cannot save picture in " + imgPath, Toast.LENGTH_LONG).show();
                    }

                    // setup image properties
                    ExifInterface newExif;
                    try {
                        int imgOrientation = ExifInterface.ORIENTATION_NORMAL;
                        switch (Cam.getCameraRotatingAngle(fcam, Camera.CameraInfo.CAMERA_FACING_BACK, camera)) {
                            case 0:
                                imgOrientation = ExifInterface.ORIENTATION_NORMAL;
                                break;
                            case 90:
                                imgOrientation = ExifInterface.ORIENTATION_ROTATE_90;
                                break;
                            case 180:
                                imgOrientation = ExifInterface.ORIENTATION_ROTATE_180;
                                break;
                            case 270:
                                imgOrientation = ExifInterface.ORIENTATION_ROTATE_270;
                                break;
                            default:
                                imgOrientation = ExifInterface.ORIENTATION_NORMAL;
                                break;
                        }
                        newExif = new ExifInterface(imgPath);
                        newExif.setAttribute(ExifInterface.TAG_DATETIME, timeNow);
                        newExif.setAttribute(ExifInterface.TAG_ORIENTATION, Integer.toString(imgOrientation));
                        newExif.saveAttributes();
                    } catch (IOException e) {
                        // do nothing
                    }

                    if (isSilent && mgr != null) {
                        // turn on the sound
                        //mgr.setStreamMute(AudioManager.STREAM_SYSTEM, false);
                        Cam.enableSound(mgr);
                    }

                    //Matrix matrix = new Matrix();
                    //matrix.postRotate(90);

                    int largerDimension = fcam.getUI().getScreenHeight()>fcam.getUI().getScreenWidth()?fcam.getUI().getScreenHeight():fcam.getUI().getScreenWidth();
                    Bitmap cb = AppLibFile.getBitmapFromPath(imgPath, largerDimension, largerDimension);

                    //Bitmap scaledBitmap = Bitmap.createScaledBitmap(cb, cb.getWidth(), cb.getHeight(),true);
                    //Bitmap rotatedBitmap = Bitmap.createBitmap(scaledBitmap , 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
                    CustomDialog customDialog = new CustomDialog(fcam, cb);
                    customDialog.DialogProcess().show();
                    fcam.current_dialog = customDialog;

                    //fcam.finishApp();
                }
            }
        });
    }

}
