#ifndef KFMElementAspectRatioExtractor_HH__
#define KFMElementAspectRatioExtractor_HH__

#include "KFMPoint.hh"
#include "KSurfaceTypes.hh"


namespace KEMField
{

/*
*
*@file KFMElementAspectRatioExtractor.hh
*@class KFMElementAspectRatioExtractor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 13:31:39 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElementAspectRatioExtractor :
    public KSelectiveVisitor<KShapeVisitor, KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>
{
  public:
    KFMElementAspectRatioExtractor(){};
    ~KFMElementAspectRatioExtractor() override
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
        fCurrentAspectRatio = -1.0;
    };
    void Visit(KRing& r) override
    {
        (void) r;
        fIsRecognized = false;
        fCurrentAspectRatio = -1.0;
    };
    void Visit(KSymmetryGroup<KTriangle>& t) override
    {
        (void) t;
        fIsRecognized = false;
        fCurrentAspectRatio = -1.0;
    };
    void Visit(KSymmetryGroup<KRectangle>& r) override
    {
        (void) r;
        fIsRecognized = false;
        fCurrentAspectRatio = -1.0;
    };
    void Visit(KSymmetryGroup<KLineSegment>& l) override
    {
        (void) l;
        fIsRecognized = false;
        fCurrentAspectRatio = -1.0;
    };
    void Visit(KSymmetryGroup<KConicSection>& c) override
    {
        (void) c;
        fIsRecognized = false;
        fCurrentAspectRatio = -1.0;
    };
    void Visit(KSymmetryGroup<KRing>& r) override
    {
        (void) r;
        fIsRecognized = false;
        fCurrentAspectRatio = -1.0;
    };

    bool IsRecognizedType() const
    {
        return fIsRecognized;
    };
    double GetAspectRatio() const
    {
        return fCurrentAspectRatio;
    };

  private:
    double TriangleAspectRatio(KFMPoint<3> P0, KFMPoint<3> P1, KFMPoint<3> P2) const;
    double RectangleAspectRatio(KFMPoint<3> P0, KFMPoint<3> P1, KFMPoint<3> P2) const;


    bool fIsRecognized;
    double fCurrentAspectRatio;
};


}  // namespace KEMField

#endif /* KFMElementAspectRatioExtractor_H__ */
