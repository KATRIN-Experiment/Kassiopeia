#ifndef KFMElectrostaticElement_HH__
#define KFMElectrostaticElement_HH__

#include "KFMBall.hh"
#include "KFMBasisData.hh"
#include "KFMIdentityPair.hh"
#include "KFMPointCloud.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticElement.hh
*@class KFMElectrostaticElement
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug 28 18:31:56 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int SpatialDimension, unsigned int BasisDimension> class KFMElectrostaticElement
{
  public:
    KFMElectrostaticElement() = default;
    ;
    virtual ~KFMElectrostaticElement() = default;
    ;

    void SetIdentityPair(const KFMIdentityPair& id)
    {
        fIDPair = id;
    };
    KFMIdentityPair GetIdentityPair() const
    {
        return fIDPair;
    };

    void SetBoundingBall(const KFMBall<SpatialDimension>& ball)
    {
        fBoundingBall = ball;
    };
    const KFMBall<SpatialDimension>& GetBoundingBall() const
    {
        return fBoundingBall;
    };

    void SetPointCloud(const KFMPointCloud<SpatialDimension>& cloud)
    {
        fPointCloud = cloud;
    };
    const KFMPointCloud<SpatialDimension>& GetPointCloud() const
    {
        return fPointCloud;
    };

    void SetBasisData(const KFMBasisData<BasisDimension>& basis)
    {
        fBasis = basis;
    };
    const KFMBasisData<BasisDimension>& GetBasisData() const
    {
        return fBasis;
    };

    void SetAspectRatio(double aspect_ratio)
    {
        fAspectRatio = aspect_ratio;
    };
    double GetAspectRatio() const
    {
        return fAspectRatio;
    };

    void SetCentroid(const KFMPoint<SpatialDimension>& centroid)
    {
        fCentroid = centroid;
    };
    const KFMPoint<SpatialDimension>& GetCentroid() const
    {
        return fCentroid;
    };


  private:
    KFMIdentityPair fIDPair;
    KFMBall<SpatialDimension> fBoundingBall;
    KFMPointCloud<SpatialDimension> fPointCloud;
    KFMBasisData<BasisDimension> fBasis;
    KFMPoint<SpatialDimension> fCentroid;
    double fAspectRatio;
};


}  // namespace KEMField


#endif /* KFMElectrostaticElement_H__ */
