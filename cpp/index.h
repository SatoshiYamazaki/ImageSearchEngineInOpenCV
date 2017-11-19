#include <cv.h>
#include <opencv2/legacy/legacy.hpp>

using namespace std;


class FeatureDB{
public:
    FeatureDB(int feature_dim)
        : m_feature_dim(feature_dim)
    {
    }
    void insertFeatures(vector<vector<float> > featureList);
    vector<int> findKNN(vector<vector<float> > queryFeatureList, int k);
private:
    int    m_feature_dim;
    CvFeatureTree*  m_ft;
};
