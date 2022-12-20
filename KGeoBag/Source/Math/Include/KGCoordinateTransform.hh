#ifndef KGCOORDINATETRANSFORM_DEF
#define KGCOORDINATETRANSFORM_DEF

#include "KThreeVector.hh"

namespace KGeoBag
{
class KGCoordinateTransform
{
  public:
    KGCoordinateTransform();

    KGCoordinateTransform(const double* p, const double* x, const double* y, const double* z);

    ~KGCoordinateTransform() = default;

    void ConvertToLocalCoords(const double* global, double* local, const bool isVec) const;
    void ConvertToGlobalCoords(const double* local, double* global, const bool isVec) const;

    void ConvertToLocalCoords(const katrin::KThreeVector& global, katrin::KThreeVector& local,
                              const bool isVec) const;
    void ConvertToGlobalCoords(const katrin::KThreeVector& local, katrin::KThreeVector& global,
                               const bool isVec) const;

  protected:
    double fP[3];  ///< Global (x,y,z) of local (0,0,0).
    double fX[3];  ///< Global (x,y,z) of local (1,0,0) unit vector.
    double fY[3];  ///< Global (x,y,z) of local (0,1,0) unit vector.
    double fZ[3];  ///< Global (x,y,z) of local (0,0,1) unit vector.
};
}  // namespace KGeoBag

#endif /* KGCOORDINATETRANSFORM_DEF */
