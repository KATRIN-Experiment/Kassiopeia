#include "KFMSurfaceToPointCloudConverter.hh"

#include "KThreeVector_KEMField.hh"

namespace KEMField
{

void KFMSurfaceToPointCloudConverter::Visit(KTriangle& t)
{
    fCurrentPointCloud.Clear();
    fIsRecognized = true;

    //get the vertices of the triangle
    KPosition p0 = t.GetP0();
    KPosition p1 = t.GetP1();
    KPosition p2 = t.GetP2();

    KFMPoint<3> pp0;
    pp0[0] = p0[0];
    pp0[1] = p0[1];
    pp0[2] = p0[2];
    KFMPoint<3> pp1;
    pp1[0] = p1[0];
    pp1[1] = p1[1];
    pp1[2] = p1[2];
    KFMPoint<3> pp2;
    pp2[0] = p2[0];
    pp2[1] = p2[1];
    pp2[2] = p2[2];

    fCurrentPointCloud.AddPoint(pp0);
    fCurrentPointCloud.AddPoint(pp1);
    fCurrentPointCloud.AddPoint(pp2);
}

void KFMSurfaceToPointCloudConverter::Visit(KRectangle& r)
{

    fCurrentPointCloud.Clear();
    fIsRecognized = true;

    //get the vertices of the rectangle
    KPosition p0 = r.GetP0();
    KPosition p1 = r.GetP1();
    KPosition p2 = r.GetP2();
    KPosition p3 = r.GetP3();

    KFMPoint<3> pp0;
    pp0[0] = p0[0];
    pp0[1] = p0[1];
    pp0[2] = p0[2];
    KFMPoint<3> pp1;
    pp1[0] = p1[0];
    pp1[1] = p1[1];
    pp1[2] = p1[2];
    KFMPoint<3> pp2;
    pp2[0] = p2[0];
    pp2[1] = p2[1];
    pp2[2] = p2[2];
    KFMPoint<3> pp3;
    pp3[0] = p3[0];
    pp3[1] = p3[1];
    pp3[2] = p3[2];

    fCurrentPointCloud.AddPoint(pp0);
    fCurrentPointCloud.AddPoint(pp1);
    fCurrentPointCloud.AddPoint(pp2);
    fCurrentPointCloud.AddPoint(pp3);
}

void KFMSurfaceToPointCloudConverter::Visit(KLineSegment& l)
{

    fCurrentPointCloud.Clear();

    fIsRecognized = true;

    //get the vertices of the line
    KPosition p0 = l.GetP0();
    KPosition p1 = l.GetP1();

    KFMPoint<3> pp0;
    pp0[0] = p0[0];
    pp0[1] = p0[1];
    pp0[2] = p0[2];
    KFMPoint<3> pp1;
    pp1[0] = p1[0];
    pp1[1] = p1[1];
    pp1[2] = p1[2];

    fCurrentPointCloud.AddPoint(pp0);
    fCurrentPointCloud.AddPoint(pp1);
}


}  // namespace KEMField
