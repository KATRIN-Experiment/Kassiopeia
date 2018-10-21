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


class KFMSurfaceToPointCloudConverter:
public KSelectiveVisitor<KShapeVisitor,KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>
{
    public:

        KFMSurfaceToPointCloudConverter(){};
        ~KFMSurfaceToPointCloudConverter(){;};

        void Visit(KTriangle& t);
        void Visit(KRectangle& r);
        void Visit(KLineSegment& l);
        void Visit(KConicSection& c){(void)c; fIsRecognized = false; fCurrentPointCloud.Clear();};
        void Visit(KRing& r){(void)r; fIsRecognized = false; fCurrentPointCloud.Clear();};
        void Visit(KSymmetryGroup<KTriangle>& t){(void)t; fIsRecognized = false; fCurrentPointCloud.Clear();};
        void Visit(KSymmetryGroup<KRectangle>& r){(void)r; fIsRecognized = false; fCurrentPointCloud.Clear();};
        void Visit(KSymmetryGroup<KLineSegment>& l){(void)l; fIsRecognized = false; fCurrentPointCloud.Clear();};
        void Visit(KSymmetryGroup<KConicSection>& c){(void)c; fIsRecognized = false; fCurrentPointCloud.Clear();};
        void Visit(KSymmetryGroup<KRing>& r){(void)r; fIsRecognized = false; fCurrentPointCloud.Clear();};

        bool IsRecognizedType() const {return fIsRecognized;};
        KFMPointCloud<3> GetPointCloud() const {return fCurrentPointCloud;};

    private:

        bool fIsRecognized;
        KFMPointCloud<3> fCurrentPointCloud;

};




}//end of KEMField

#endif /* KFMSurfaceToPointCloudConverter_H__ */
