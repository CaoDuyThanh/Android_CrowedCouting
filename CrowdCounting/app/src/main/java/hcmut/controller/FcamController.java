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
import org.opencv.core.Point;
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
    static {

    }

    private CountingMain fcam;

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

    private Mat multiplyByScala(Mat oldMatrix, double scalar){
        return oldMatrix.mul(oldMatrix.clone(), scalar);
    }

    Mat getGaborKernel( double w, double theta, double sigma,
                        int filterSize, String filterType )
    {
        Mat kernel = new Mat(filterSize, filterSize, CvType.CV_32F);

        double xr = -Math.floor(filterSize / 2.0) / filterSize;
        double yr = -xr;
        double step = 1.0 / filterSize;
        double xrTemp, yrTemp;
        double rotation;
        double bandpass;
        double filterRadius = -Math.floor(filterSize / 2.0);
        double xg, yg, k, gauss;
        double sigmaScale = sigma * (filterSize / 2.0);
        double A = 1 / (2 * Math.PI * sigmaScale * sigmaScale);
        for (int row  = 0; row < filterSize; row++){
            yrTemp = yr - row * step;
            for (int col = 0; col < filterSize; col++){
                xrTemp = xr + col * step;
                rotation = xrTemp * Math.cos(theta) + yrTemp * Math.sin(theta);

                if (filterType.equals("even")){
                    bandpass = Math.cos(2 * Math.PI * w * rotation);
                }
                else{
                    bandpass = Math.sin(2 * Math.PI * w * rotation);
                }

                xg = filterRadius + row;
                yg = - filterRadius - col;

                k = -1 * (( xg * xg + yg * yg) / (2 * sigmaScale * sigmaScale));
                gauss = A * Math.exp(k);

                kernel.put(row, col, gauss * bandpass);
            }
        }

        return kernel;
    }

    Mat histEq(Mat a, int l){
        Mat result = new Mat(new Size(a.cols(), a.rows()), CvType.CV_8UC1);
        int[] f = new int[256];
        double[] pdf = new double[256];
        int n = a.rows() * a.cols();

        for (int row = 0; row < a.rows(); row++){
            for (int col = 0; col < a.cols(); col++){
                int value = (int)a.get(row, col)[0];
                f[value]++;
                pdf[value] = f[value] * 1.0 / (double)n;
            }
        }

        int sum = 0;
        int[] cum = new int[256];
        double[] cdf = new double[256];
        int[] out = new int[256];
        for (int i = 0; i < 256; i++) {
            sum += f[i];
            cum[i] = sum;
            cdf[i] = cum[i] * 1.0 / (double)n;
            out[i] = (int)Math.round(cdf[i] * l);
        }

        for (int row = 0; row < a.rows(); row++){
            for (int col = 0; col < a.cols(); col++){
                int value = (int)a.get(row, col)[0];
                result.put(row, col, out[value]);
            }
        }

        return result;
    }

    public String extractFeature(Bitmap input){
        // resize to desired size
        int kernel_size = Const.KERNEL_SIZE;
        int width = Const.IMG_RESIZE_WIDTH;
        int height = Const.IMG_RESIZE_HEIGHT;

        // convert to Mat (OpenCV datatype)
        Mat img = new Mat();
        Utils.bitmapToMat(input, img);

        // convert to grayscale
        //Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2GRAY);
        printMat(img, 31, 31);

        // resize img
        Imgproc.resize(img, img, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
        printMat(img, 31, 31);

        // histogram equalization
        //Imgproc.equalizeHist(img, img);
        //img = histEq(img, 255);
        //printMat(img, 31, 31);
        // Transfer data from Mat type to MWNumericArray


/*        MWNumericArray imgMat = new MWNumericArray();

        GaborFeatureHandler gaborFeatureHandler = null;
        try {
            gaborFeatureHandler = new GaborFeatureHandler();
            Object[] result = gaborFeatureHandler.ExtractFeature(1, imgMat);
        } catch (MWException e) {
            System.out.println(e);
        }
*/



        // generate Gabor kernel
        // params
        double sigma;  // tmp value
        double[] theta = new double[]{0, Math.PI / 4, Math.PI / 2, 3 * Math.PI / 4};
        double scaleFactor = (double)Const.IMG_RESIZE_WIDTH / kernel_size;
        double[] w = new double[]{2.0 / scaleFactor,
                                  4.0 / scaleFactor,
                                  8.0 / scaleFactor,
                                  16.0 / scaleFactor,
                                  32.0 / scaleFactor,
                                  64.0 / scaleFactor};
        double gamma = 1.0f;
        double psi = 0.0f;

        double[] vector = new double[2 * theta.length * w.length];

        int count = 0;
        for (int l = 0; l < w.length; l++) {
            for (int t = 0; t < theta.length; t++) {

                Mat gaborKernel = getGaborKernel(w[l], theta[t], 1 / w[l], kernel_size ,"even");

                Mat imgAtKernel = img.clone();
                imgAtKernel.convertTo(imgAtKernel, CvType.CV_32F);

                Imgproc.filter2D(imgAtKernel, imgAtKernel, CvType.CV_32F, gaborKernel, new Point(-1, -1), 0, 2);        // BORDER_REFLECT = 2

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
                vector[count + 1] = AppLibGeneral.round(variance, Const.ROUND_DIGITS);
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
        //return double2string(vector);

        String featureJsonObject = "";
        try {
            featureJsonObject = toJSON(vector);
        }catch (JSONException e){
            // Catch the exception and handle it
        }

        return featureJsonObject;
    }

    public void printMat(Mat mat, int width, int height){
        System.out.println();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                System.out.print(mat.get(i, j)[0]);
                System.out.print("   ");
            }
            System.out.println();
        }
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
