package com.example.imageclassification;

import static android.widget.Toast.LENGTH_SHORT;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Matrix;
import android.os.Bundle;
import androidx.annotation.Nullable;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.text.Editable;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;


import com.example.imageclassification.ml.Model;

import com.example.imageclassification.ml.Modell;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result,showarray,corr_img, t1;
    int imageSize = 224;

    //Double sg_th = 0.0;

    final Context c = this;

    private ProgressBar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //pgbar
        progressBar = (ProgressBar) findViewById (R.id.progressBar);
        t1 = findViewById(R.id.t1);
        //

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        showarray = findViewById(R.id.showarray);
        corr_img = findViewById(R.id.correctImage);

        imageView = findViewById(R.id.imageView);


        /* Set an EditText view to get user input---------------------------------------------------//

        LayoutInflater layoutInflaterAndroid = LayoutInflater.from(c);
        View mView = layoutInflaterAndroid.inflate(R.layout.user_input_dialog_box, null);
        AlertDialog.Builder alertDialogBuilderUserInput = new AlertDialog.Builder(c);
        alertDialogBuilderUserInput.setView(mView);

        final EditText input = (EditText) mView.findViewById(R.id.userInputDialog);
        alertDialogBuilderUserInput
                .setCancelable(false)
                .setPositiveButton("Done", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialogBox, int id) {

                       sg_th = Double.valueOf(input.getText().toString());
                        // ToDo get user input here
                    }
                })

                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialogBox, int id) {
                                dialogBox.cancel();
                            }
                        });

        AlertDialog alertDialogAndroid = alertDialogBuilderUserInput.create();
        alertDialogAndroid.show();

        //------------------------------------------------------------------------------------------*/

        camera.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                }
                else
                {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                cameraIntent.setType("image/*");
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    //-----------------------------------------------------CHECK-IMAGE-----------------------------------------------------//
    public void checkImage(Bitmap image)
    {
        //final boolean[] flag = {false};
        progressBar.setVisibility(View.VISIBLE);
        t1.setVisibility(View.VISIBLE);

        Handler handler = new Handler();
        handler.postDelayed(new Runnable()
        {
            public void run()
            {
                progressBar.setVisibility(View.GONE);
                t1.setVisibility(View.GONE);

                showarray.setText("");
                result.setText("");
                corr_img.setText("");

                try {
                    Modell model = Modell.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                    ByteBuffer byteBuffer1 = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
                    byteBuffer1.order(ByteOrder.nativeOrder());

                    int[] intValues = new int[imageSize * imageSize];
                    image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
                    int pixel = 0;
                    //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
                    for(int i = 0; i < imageSize; i ++){
                        for(int j = 0; j < imageSize; j++){
                            int val = intValues[pixel++];
                            //RGB
                            byteBuffer1.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f)); //Red
                            byteBuffer1.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f)); //Green
                            byteBuffer1.putFloat((val & 0xFF) * (1.f / 255.f)); //Blue
                        }
                    }

                    inputFeature0.loadBuffer(byteBuffer1);

                    // Runs model inference and gets result.
                    Modell.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] confidences = outputFeature0.getFloatArray();
                    // find the index of the class with the biggest confidence.
                    int maxPos = 0;
                    float maxConfidence = 0;
                    for (int i = 0; i < confidences.length; i++) {
                        if (confidences[i] > maxConfidence) {
                            maxConfidence = confidences[i];
                            maxPos = i;
                        }
                    }

                    String[] classes = {"SugarCane", "Unknown"};
                    String checker = classes[maxPos];

                    if(checker.equals("SugarCane") && maxConfidence >= 0.85) //Then is SugarCane
                    {
                        classifyImage(image);
                        corr_img.setText("Is SugarCane Confidence: "+maxConfidence+"%");
                    }
                    else if(checker.equals("Unknown") && maxConfidence >= 0.85) //Not SugarCane
                    {
                        corr_img.setText("Not SugarCane Confidence: "+maxConfidence+"%");
                        //
                        showarray.setText("");
                        result.setText("Classified As:\n{Unknown}");
                    }
                    else
                    {
                        corr_img.setText("Cannot Classify Image!");
                    }

                    // Releases model resources if no longer used.
                    model.close();
                }
                catch (IOException e)
                {
                    // TODO Handle the exception
                }
            }
        }, 3000); // 3000 milliseconds delay
    }
    //--------------------------------------------------------CHECK IMAGE END------------------------------------------------//

    //---------------------------------------------------------------Disease Predictor1------------------------------------------
    public void classifyImage(Bitmap image)
    {
        progressBar.setVisibility(View.GONE);
        t1.setVisibility(View.GONE);

        showarray.setText("");
        result.setText("");
        corr_img.setText("");

        try
        {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++];
                    //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f)); //Red
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f)); //Green
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f)); //Blue
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();


            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Healthy", "RedRot", "RedRust"};//, "Unknown"

            //Confidence Threshold
            if(maxConfidence >= 0.70) //Then is SugarCane [You can set your own threshold]
            {
                result.setText("Classified As:\n{"+classes[maxPos]+"}");

                String s="";
                for(int i=0; i<classes.length; i++)
                {
                    s += String.format("%s: %.1f%%\n",classes[i], confidences[i] * 100);
                }
                showarray.setText(s);
            }
            else
            {
                //result.setText("Cannot Classify\n[Low Confidence]");
                showarray.setText("");
                corr_img.setText("");
                result.setText("Classified As:\n{Unknown}");
            }
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }//-----------------------------------------------------PREDICTIONS1 END------------------------------------------

    //++++++++++++++++++++++++++++++++++++++++++++RESIZE HUGE IMAGE++++++++++++++++++++++++++++++++++++++++++++++++++
    private static Bitmap resize(Bitmap image, int maxWidth, int maxHeight) {
        if (maxHeight > 0 && maxWidth > 0) {
            int width = image.getWidth();
            int height = image.getHeight();
            float ratioBitmap = (float) width / (float) height;
            float ratioMax = (float) maxWidth / (float) maxHeight;

            int finalWidth = maxWidth;
            int finalHeight = maxHeight;
            if (ratioMax > ratioBitmap) {
                finalWidth = (int) ((float) maxHeight * ratioBitmap);
            } else {
                finalHeight = (int) ((float) maxWidth / ratioBitmap);
            }
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true);
            return image;
        } else {
            return image;
        }
    }
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++END

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data)
    {
        if(resultCode == RESULT_OK)
        {
            if(requestCode == 3) //IMAGE TAKEN FROM CAMERA
            {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

                checkImage(image);
            }
            else //IMAGE TAKEN FROM GALLERY
            {
                Uri dat = data.getData();
                Bitmap image = null, resizeBitmap = null;
                try
                {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);

                    if(image.getWidth() > 4000 && image.getHeight() > 3000) //DEAL WITH HIGH RESOLUTION PICTURE
                    {
                        resizeBitmap = resize(image, image.getWidth() / 2, image.getHeight() / 2);

                        imageView.setImageBitmap(resizeBitmap);

                        resizeBitmap = Bitmap.createScaledBitmap(resizeBitmap, imageSize, imageSize, false);

                        checkImage(resizeBitmap);
                    }
                    else if (image.getWidth() < 200 && image.getHeight() < 200)//bad resolution
                    {
                        corr_img.setText("OPPS! Bad Image Quality");
                        showarray.setText("");
                        result.setText("");
                    }
                    else
                    {
                        imageView.setImageBitmap(image);

                        image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

                        checkImage(image);
                    }
                }
                catch (IOException e)
                {
                    e.printStackTrace();
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}