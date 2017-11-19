#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>

using namespace std;

const int DIM_VECTOR = 128;  // 128次元ベクトル

/**
 * SURF情報をファイルに出力
 * @param[in]   filename            SURFを保存するファイル名
 * @param[in]   imageKeypoints      SURFキーポイント情報
 * @param[in]   imageDescriptors    SURF特徴ベクトル情報
 * @return なし
 */
void writeSURF(const char* filename, CvSeq* imageKeypoints, CvSeq* imageDescriptors) {
    fstream fout;

    fout.open(filename, ios::out);
    if (!fout.is_open()) {
        cerr << "cannot open file: " << filename << endl;
        return;
    }

    // 1行目はキーポイント数と特徴量の次元数を書き込む
    fout << imageKeypoints->total << ' ' << DIM_VECTOR << endl;

    // 2行目からキーポイント情報と特徴ベクトルを書き込む
    for (int i = 0; i < imageKeypoints->total; i++) {
        CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(imageKeypoints, i);
        float* descriptor = (float*)cvGetSeqElem(imageDescriptors, i);
        // キーポイント情報（X座標, Y座標, サイズ, ラプラシアン）を書き込む
        fout << point->pt.x << ' ' << point->pt.y << ' ' << point->size << ' ' << point->laplacian << ' ';
        // 特徴ベクトルを書き込む
        for (int j = 0; j < DIM_VECTOR; j++) {
            fout << descriptor[j] << ' ';
        }
        fout << endl;
    }

    fout.close();
}

int main(int argc, char** argv) {
    const char* imageFile = argc == 3 ? argv[1] : "image/accordion_image_0001.jpg";
    const char* surfFile  = argc == 3 ? argv[2] : "image/accordion_image_0001.surf";

    // SURF抽出用に画像をグレースケールで読み込む
    IplImage* grayImage = cvLoadImage(imageFile, CV_LOAD_IMAGE_GRAYSCALE);
    if (!grayImage) {
        cerr << "cannot find image file: " << imageFile << endl;
        return -1;
    }

    // キーポイント描画用にカラーでも読み込む
    IplImage* colorImage = cvLoadImage(imageFile, CV_LOAD_IMAGE_COLOR);
    if (!colorImage) {
        cerr << "cannot find image file: " << imageFile << endl;
        return -1;
    }

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* imageKeypoints = 0;
    CvSeq* imageDescriptors = 0;
    CvSURFParams params = cvSURFParams(500, 1);

    // 画像からSURFを取得
    cvExtractSURF(grayImage, 0, &imageKeypoints, &imageDescriptors, storage, params);
    cout << "Image Descriptors: " << imageDescriptors->total << endl;

    // SURFをファイルに出力
    writeSURF(surfFile, imageKeypoints, imageDescriptors);

    // 画像にキーポイントを描画
    for (int i = 0; i < imageKeypoints->total; i++) {
        CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(imageKeypoints, i);
        CvPoint center;  // キーポイントの中心座標
        int radius;      // キーポイントの半径
        center.x = cvRound(point->pt.x);
        center.y = cvRound(point->pt.y);
        radius = cvRound(point->size * 1.2 / 9.0 * 2.0);
        cvCircle(colorImage, center, radius, cvScalar(0,255,255), 1, 8, 0);
    }

    cvNamedWindow("SURF");
    cvShowImage("SURF", colorImage);
    cvWaitKey(0);

    // 後始末
    cvReleaseImage(&grayImage);
    cvReleaseImage(&colorImage);
    cvClearSeq(imageKeypoints);
    cvClearSeq(imageDescriptors);
    cvReleaseMemStorage(&storage);
    cvDestroyAllWindows();

    return 0;
}
