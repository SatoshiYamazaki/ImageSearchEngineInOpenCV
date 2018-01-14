#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <map>
#include "index.h"

using namespace std;

const int DIM = 128;
const int SURF_PARAM = 400;

vector<vector<float> > extractSURF(IplImage* queryImage);
bool loadObjectId(const char *filename, map<int, string>& id2name);
bool loadDescription(const char *filename, vector<int> &labels, vector<vector<float> > &featureList);

int main(int argc, char** argv) {
    double tt = (double)cvGetTickCount();
    const char* OBJID_FILE = "object.txt";
    const char* IMAGE_DIR = "caltech101_10";
    const char* DESC_FILE = "description.txt";

    cout << "Create objectID->objectName hash... " << flush;
    map<int, string> id2name;
    if (!loadObjectId(OBJID_FILE, id2name)) {
        cerr << "cannot load object id file" << endl;
        return 1;
    }
    cout << "OK" << endl;

    cout << "Loading object database... " << flush;
    FeatureDB fdb(DIM);
    vector<int> labels;     // KeyPoint labels
    vector<vector<float> > featureList;
    if (!loadDescription(DESC_FILE, labels, featureList)) {
        cerr << "cannot load description file" << endl;
        return 1;
    }
    cout << "OK" << endl;

    fdb.insertFeatures(featureList);
    cout << "# of objects: " << id2name.size() << endl;
    tt = (double)cvGetTickCount() - tt;
    cout << "Loading Models Time = " << tt / (cvGetTickFrequency() * 1000.0) << "ms" << endl;

    while (1) {
        // input query
        char input[1024];
        cout << "query? > ";
        cin >> input;

        char queryFile[1024];
        snprintf(queryFile, sizeof queryFile, "%s/%s", IMAGE_DIR, input);
        cout << queryFile << endl;
        tt = (double)cvGetTickCount();

        // Load query image
        IplImage *queryImage = cvLoadImage(queryFile, CV_LOAD_IMAGE_GRAYSCALE);
        if (queryImage == NULL) {
            cerr << "cannot load image file: " << queryFile << endl;
            continue;
        }

        // vote
        int numObjects = (int)id2name.size();
        int votes[numObjects];
        for (int i = 0; i < numObjects; i++) {
            votes[i] = 0;
        }

        // extract queryFeatureList
        vector<vector<float> > queryFeatureList = extractSURF(queryImage);

        // Finf K nearest neighbors (k=1)
        std::vector<int> v = fdb.findKNN(queryFeatureList,1);
        for (int i = 0; i < v.size(); i++) {
            votes[labels[v[i]]]++;
        }
        int maxId = -1;
        int maxVal = -1;
        for (int i = 0; i < numObjects; i++) {
            if (votes[i] > maxVal) {
                maxId = i;
                maxVal = votes[i];
            }
        }

        string name = id2name[maxId];
        cout << "Result: " << name << endl;

        tt = (double)cvGetTickCount() - tt;
        cout << "Recognition Time = " << tt / (cvGetTickFrequency() * 1000.0) << "ms" << endl;

        cvReleaseImage(&queryImage);
        cvDestroyAllWindows();
    }
    return 0;
}

/**
 * Extract SURF features
 *
 * @param[in]  queryImage  IplImage
 *
 * @return SURF features list
 */

vector<vector<float> > extractSURF(IplImage* queryImage){
  // Extract SURF features
  CvSeq *queryKeypoints = 0;
  CvSeq *queryDescriptors = 0;
  CvMemStorage *storage = cvCreateMemStorage(0);
  CvSURFParams params = cvSURFParams(SURF_PARAM, 1);
  cvExtractSURF(queryImage, 0, &queryKeypoints, &queryDescriptors, storage, params);
  cout << "# of key points in query image: " << queryKeypoints->total << endl;

  // extract queryFeatureList
  vector<vector<float> > queryFeatureList;
  for (int i = 0; i < queryKeypoints->total; i++) {
      CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(queryKeypoints, i);
      float* descriptor = (float*)cvGetSeqElem(queryDescriptors, i);
      vector<float> tmpFeature;
      for (int j = 0; j < DIM; j++) {
          tmpFeature.push_back(descriptor[j]);
      }
      queryFeatureList.push_back(tmpFeature);
  }
  cvClearSeq(queryKeypoints);
  cvClearSeq(queryDescriptors);
  cvReleaseMemStorage(&storage);
  return queryFeatureList;
}

/**
 * Create objectID->objectName map
 *
 * @param[in]  filename  file name for objectID->objectName correspondance
 * @param[out] id2name   objectID->objectName map
 *
 * @return true(success) or false(failed)
 */
bool loadObjectId(const char *filename, map<int, string>& id2name) {
    ifstream objFile(filename);
    if (objFile.fail()) {
        cerr << "cannot open file: " << filename << endl;
        return false;
    }

    string line;
    while (getline(objFile, line, '\n')) {
        vector<string> ldata;
        istringstream ss(line);
        string s;
        while (getline(ss, s, '\t')) {
            ldata.push_back(s);
        }

        int objId = atol(ldata[0].c_str());
        string objName = ldata[1];
        id2name.insert(map<int, string>::value_type(objId, objName));
    }
    objFile.close();
    return true;
}

/**
 * Load feature information (SURF features)
 *
 * @param[in]  filename  filename for feature vectors
 * @param[out] labels    ObjectIDs for feature vectors
 * @param[out] objMat    Feature vector matrix
 *
 * @return true(success) or false(failed)
 */
bool loadDescription(const char *filename, vector<int> &labels, vector<vector<float> > &featureList) {
    ifstream descFile(filename);
    if (descFile.fail()) {
        cerr << "cannot open file: " << filename << endl;
        return false;
    }
    string line;
    descFile.clear();
    descFile.seekg(0);

    while (getline(descFile, line, '\n')) {
        vector<string> ldata;
        istringstream ss(line);
        string s;
        while (getline(ss, s, '\t')) {
            ldata.push_back(s);
        }
        int objId = atol(ldata[0].c_str());
        labels.push_back(objId);
        vector<float> keys;
        for (int j = 0; j < DIM; j++) {
            keys.push_back(atof(ldata[j+2].c_str()));  // 特徴ベクトルはldata[2]から
        }
        featureList.push_back(keys);
    }
    descFile.close();
    return true;
}
