#include "searchdb.h"

void FeatureDB::insertFeatures(vector<vector<float> > featureList){
  CvMat* objMat;
  objMat = cvCreateMat(featureList.size(), m_feature_dim, CV_32FC1);
  for (size_t l=0; l<featureList.size(); l++){
    for (int j = 0; j < m_feature_dim; j++) {
        float val = featureList[l][j];
        CV_MAT_ELEM(*objMat, float, l, j) = val;
    }
  }
  cout << "Now indexing database... " << flush;
  m_ft = cvCreateKDTree(objMat);
  cout << "OK" << endl;
  cout << "# of key points: " << objMat->rows << endl;
};

vector<int> FeatureDB::findKNN(vector<vector<float> > queryFeatureList, int k){
  CvMat* queryMat = cvCreateMat(queryFeatureList.size(), m_feature_dim, CV_32FC1);
  for (size_t l=0; l<queryFeatureList.size(); l++){
    for (int j = 0; j < m_feature_dim; j++) {
        float val = queryFeatureList[l][j];
        CV_MAT_ELEM(*queryMat, float, l, j) = val;
    }
  }

  CvMat* indices = cvCreateMat(queryFeatureList.size(), k, CV_32SC1);
  CvMat* dists = cvCreateMat(queryFeatureList.size(), k, CV_64FC1);
  cvFindFeatures(m_ft, queryMat, indices, dists, k, 250);
  std::vector<int> v;
  for (int i = 0; i < indices->rows; i++) {
      int idx = CV_MAT_ELEM(*indices, int, i, 0);
      v.push_back(idx);
  }

  return v;
}
