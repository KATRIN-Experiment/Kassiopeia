#ifndef KFMSurfaceToPointCloudConverter_HH__
#define KFMSurfaceToPointCloudConverter_HH__

#include "KFMPointCloud.hh"
#include "KSurfaceTypes.hh"


namespace KEMField
{

/*
*
*@file KFMSurfaceToPointCloudConverter.hh
*@class KFMSurfaceToPointCloudConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 13:31:39 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMSurfaceToPointCloudConverter :
    public KSelectiveVisitor<KShapeVisitor, KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>
{
  public:
    KFMSurfaceToPointCloudConverter(){};
    ~KFMSurfaceToPointCloudConverter() override
    {
        ;
    };

    void Visit(KTriangle& t) override;
    void Visit(KRectangle& r) override;
    void Visit(KLineSegment& l) override;
    void Visit(KConicSection& c) override
    {
        (void) c;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };
    void Visit(KRing& r) override
    {
        (void) r;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };
    void Visit(KSymmetryGroup<KTriangle>& t) override
    {
        (void) t;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };
    void Visit(KSymmetryGroup<KRectangle>& r) override
    {
        (void) r;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };
    void Visit(KSymmetryGroup<KLineSegment>& l) override
    {
        (void) l;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };
    void Visit(KSymmetryGroup<KConicSection>& c) override
    {
        (void) c;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };
    void Visit(KSymmetryGroup<KRing>& r) override
    {
        (void) r;
        fIsRecognized = false;
        fCurrentPointCloud.Clear();
    };

    bool IsRecognizedType() const
    {
        return fIsRecognized;
    };
    KFMPointCloud<3> GetPointCloud() const
    {
        return fCurrentPointCloud;
    };

  private:
    bool fIsRecognized;
    KFMPointCloud<3> fCurrentPointCloud;
};


}  // namespace KEMField

#endif /* KFMSurfaceToPointCloudConverter_H__ */
