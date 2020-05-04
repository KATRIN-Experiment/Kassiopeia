#ifndef KROTATION_H_
#define KROTATION_H_

#include "KThreeMatrix.hh"

#include <vector>
using std::vector;

namespace KGeoBag
{

class KRotation : public KThreeMatrix
{
  public:
    KRotation();
    KRotation(const KRotation& aRotation);
    KRotation& operator=(const KRotation& other) = default;
    ~KRotation() override;

    KRotation& operator=(const KThreeMatrix& aMatrix);

    void SetIdentity();
    void SetAxisAngle(const KThreeVector& anAxis, const double& anAngle);
    void SetAxisAngleInDegrees(const KThreeVector& anAxis, const double& anAngle);
    void SetEulerAngles(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerAnglesInDegrees(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerAngles(const vector<double>& anArray);
    void SetEulerAnglesInDegrees(const vector<double>& anArray);
    void SetEulerZYZAngles(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerZYZAnglesInDegrees(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerZYZAngles(const vector<double>& anArray);
    void SetEulerZYZAnglesInDegrees(const vector<double>& anArray);
    void SetRotatedFrame(const KThreeVector& x, const KThreeVector& y, const KThreeVector& z);
};

}  // namespace KGeoBag

#endif
