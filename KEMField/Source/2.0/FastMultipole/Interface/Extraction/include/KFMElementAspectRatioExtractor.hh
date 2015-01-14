#ifndef KFMElementAspectRatioExtractor_HH__
#define KFMElementAspectRatioExtractor_HH__

#include "KSurfaceTypes.hh"
#include "KFMPoint.hh"


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


class KFMElementAspectRatioExtractor:
public KSelectiveVisitor<KShapeVisitor,KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>
{
    public:

        KFMElementAspectRatioExtractor(){};
        ~KFMElementAspectRatioExtractor(){;};

        void Visit(KTriangle& t);
        void Visit(KRectangle& r);
        void Visit(KLineSegment& l);
        void Visit(KConicSection& c){(void)c; fIsRecognized = false; fCurrentAspectRatio = -1.0;};
        void Visit(KRing& r){(void)r; fIsRecognized = false; fCurrentAspectRatio = -1.0;};
        void Visit(KSymmetryGroup<KTriangle>& t){(void)t; fIsRecognized = false; fCurrentAspectRatio = -1.0;};
        void Visit(KSymmetryGroup<KRectangle>& r){(void)r; fIsRecognized = false; fCurrentAspectRatio = -1.0;};
        void Visit(KSymmetryGroup<KLineSegment>& l){(void)l; fIsRecognized = false; fCurrentAspectRatio = -1.0;};
        void Visit(KSymmetryGroup<KConicSection>& c){(void)c; fIsRecognized = false; fCurrentAspectRatio = -1.0;};
        void Visit(KSymmetryGroup<KRing>& r){(void)r; fIsRecognized = false; fCurrentAspectRatio = -1.0;};

        bool IsRecognizedType() const {return fIsRecognized;};
        double GetAspectRatio() const {return fCurrentAspectRatio;};

    private:

        double TriangleAspectRatio(KFMPoint<3> P0, KFMPoint<3> P1, KFMPoint<3> P2) const;
        double RectangleAspectRatio(KFMPoint<3> P0, KFMPoint<3> P1, KFMPoint<3> P2) const;


        bool fIsRecognized;
        double fCurrentAspectRatio;

};




}//end of KEMField

#endif /* KFMElementAspectRatioExtractor_H__ */
