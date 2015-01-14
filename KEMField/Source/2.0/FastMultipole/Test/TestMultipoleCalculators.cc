#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cmath>

#include "KFMPoint.hh"
#include "KFMPointCloud.hh"

#include "KEMThreeVector.hh"

#include "KVMPathIntegral.hh"
#include "KVMLineIntegral.hh"
#include "KVMSurfaceIntegral.hh"
#include "KVMFluxIntegral.hh"

#include "KVMField.hh"
#include "KVMFieldWrapper.hh"

#include "KVMLineSegment.hh"
#include "KVMTriangularSurface.hh"
#include "KVMRectangularSurface.hh"

#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"
#include "KFMElectrostaticMultipoleCalculatorNumeric.hh"



using namespace KEMField;


class FieldTestCase
{
    public:
        FieldTestCase():fOrigin(0,0,0){;};
        virtual ~FieldTestCase(){;};

        void identity(const double* /*in*/, double* out) const
        {
            out[0] = 1.0;
        };

        //function which takes a point in 3d an returns a 6d array
        void functionA(const double* in, double* out) const
        {
            out[0] = in[0]*in[0];
            out[1] = in[0]*in[1];
            out[2] = in[0]*in[2];
            out[3] = in[1]*in[1];
            out[4] = in[1]*in[2];
            out[5] = in[2]*in[2];
        }

        void dist(const double* in, double* out) const
        {
            //compute distance to origin
            double dx,dy,dz;
            dx = in[0] - fOrigin.X();
            dy = in[1] - fOrigin.Y();
            dz = in[2] - fOrigin.Z();
            out[0] = dx*dx + dy*dy + dz*dz;
        };

        void radius_vec(const double* dom, double* range) const
        {
            range[0] = dom[0] - fOrigin.X();
            range[1] = dom[1] - fOrigin.Y();
            range[2] = dom[2] - fOrigin.Z();
        }

        void SetOrigin(KEMThreeVector origin){fOrigin = origin;};

    private:

        KEMThreeVector fOrigin;

};

int main(int argc, char* argv[])
{

    int type_select = 1;

    if(argc != 2)
    {
        std::cout<<"select type: 1(triangle), 2(rectangle), or 3 (wire)"<<std::endl;
        return 1;
    }

    std::stringstream ss;
    ss.clear();
    ss << argv[1];
    ss >> type_select;

    int degree = 3;

    //geometry for the electrodes
    double z_offset = 0.0; //1.1;

    //triangle descriptors
//    KEMThreeVector v1(1.,0.,0.);
//    KEMThreeVector v2(-0.5, std::sqrt(3.)/2.0, z_offset);
//    KEMThreeVector v3(-0.5, -1.0*std::sqrt(3.0)/2.0, z_offset);

    KEMThreeVector v1(-1.,0.,0.);
    KEMThreeVector v2(0.5, std::sqrt(3.)/2.0, z_offset) ;
    KEMThreeVector v3(0.5, -1.0*std::sqrt(3.0)/2.0, z_offset);

    KEMThreeVector axis1 = (v2 - v1).Unit();
    KEMThreeVector axis2 = (v3 - v1).Unit();
    axis1 = axis1.Unit();
    axis2 = axis2.Unit();
    double Tsize = 1;
    double TriN1[3] = {axis1[0], axis1[1], axis1[2]}; //direction of side 1
    double TriN2[3] = {axis2[0], axis2[1], axis2[2]}; //direction of side 2
    double TriA = Tsize*(v2 - v1).Magnitude();//1.732038; //length of side 1
    double TriB = Tsize*(v3 - v1).Magnitude();//1.732038; //lenght of side 2
    double TriP[3] = {0.0,0.0,0.};//{v1.X(), v1.Y(), v1.Z()};// {Tsize,0,z_offset}; //corner1
//    double TriP[3] =  {Tsize,0,z_offset}; //corner1
    KEMThreeVector TriP2;//corner2
    KEMThreeVector TriP3; //corner3
    TriP2.SetComponents(TriP[0], TriP[1], TriP[2]);
    TriP2 += TriA*axis1;
    TriP3.SetComponents(TriP[0], TriP[1], TriP[2]);
    TriP3 += TriB*axis2;

    //geometry descriptors for the rectangle used
    //sides parallel to x and y axes respectively

    double RecA = 1.1010101; //length of side 1
    double RecB = 1.0; //length of side 2
    double RecP[3] = {0,0,0}; //corner
    double RecN1[3] = {1.0,0,0}; //direction of side 1
    double RecN2[3] = {0,1.0,0}; //direction of side 2

    //geometry for the wire electrode used
    double WireLength = 1.1010101;
//    double WireDiameter = 0.01;
    double WireStartPoint[3] = {0,0,0};
    KEMThreeVector dir(1.,2.,3.); dir = dir.Unit();
    double WireDirection[3] = {dir[0], dir[1], dir[2]};
    double WireEndPoint[3] = {WireStartPoint[0]+WireLength*WireDirection[0],
                                WireStartPoint[1]+WireLength*WireDirection[1],
                                WireStartPoint[2]+WireLength*WireDirection[2]};

    //set up line segment, and triangle, rectangle surfaces
    KVMLineSegment* line = new KVMLineSegment();
    line->SetAll(WireStartPoint, WireEndPoint);
    line->Initialize();

    KVMTriangularSurface* triangle = new KVMTriangularSurface();
    triangle->SetAll(TriP, TriN1, TriN2, TriA, TriB);
    triangle->Initialize();

    KVMRectangularSurface* rectangle = new KVMRectangularSurface();
    rectangle->SetAll(RecP, RecN1, RecN2, RecA, RecB);
    rectangle->Initialize();

    //extract their vertices
    std::vector< KFMPoint<3> > triangleVertices;    triangleVertices.resize(3);
    std::vector< KFMPoint<3> > rectangleVertices;   rectangleVertices.resize(4);
    std::vector< KFMPoint<3> > wireVertices;    wireVertices.resize(2);

    triangle->GetVertices( (triangleVertices[0]), (triangleVertices[1]), (triangleVertices[2]) );
    rectangle->GetVertices( (rectangleVertices[0]), (rectangleVertices[1]), (rectangleVertices[2]), (rectangleVertices[3]) );
    line->GetVertices( (wireVertices[0]), (wireVertices[1]) );

    //insert into point clouds
    KFMPointCloud<3> triangleCloud; triangleCloud.SetPoints(&triangleVertices);
    KFMPointCloud<3> rectangleCloud; rectangleCloud.SetPoints(&rectangleVertices);
    KFMPointCloud<3> wireCloud; wireCloud.SetPoints(&wireVertices);


    //now lets make the multipole calculators
    KFMElectrostaticMultipoleCalculatorAnalytic* aCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();
    aCalc->SetDegree(degree);

    KFMElectrostaticMultipoleCalculatorNumeric* nCalc = new KFMElectrostaticMultipoleCalculatorNumeric();
    nCalc->SetDegree(degree);
    nCalc->SetNumberOfQuadratureTerms(8);

    double origin[3];
    KEMThreeVector origin_vec(0.,0.,-1.0);
    origin_vec = origin_vec.Unit();
    double mag = 1.0;
    origin_vec *= mag;

//    origin[0] = origin_vec[0];
//    origin[1] = origin_vec[1];//0.01;
//    origin[2] = origin_vec[2];
    origin[0] = 0.0;
    origin[1] = 1e-5;
    origin[2] = 1.0 - 1e-5;

    KFMScalarMultipoleExpansion aExpan;
    KFMScalarMultipoleExpansion nExpan;
    aExpan.SetDegree(degree);
    nExpan.SetDegree(degree);

    switch(type_select)
    {
        case 1:
            //aCalc->ConstructExpansion(origin, &triangleCloud, &aExpan);
            aCalc->ConstructExpansion(origin, &triangleCloud, &aExpan);
            nCalc->ConstructExpansion(origin, &triangleCloud, &nExpan);
        break;
        case 2:
            aCalc->ConstructExpansion(origin, &rectangleCloud, &aExpan);
            nCalc->ConstructExpansion(origin, &rectangleCloud, &nExpan);
        break;
        case 3:
            aCalc->ConstructExpansion(origin, &wireCloud, &aExpan);
            nCalc->ConstructExpansion(origin, &wireCloud, &nExpan);
        break;
    }



    std::vector< std::complex<double> > aMom;
    std::vector< std::complex<double> > nMom;

    aExpan.GetMoments(&aMom);
    nExpan.GetMoments(&nMom);


    for(unsigned int i=0; i<aMom.size(); i++)
    {
        std::cout<<"difference @ "<<i<<" = "<<std::abs(aMom[i] - nMom[i])<<std::endl;
    }

    for(unsigned int i=0; i<aMom.size(); i++)
    {
        std::cout<<"aMom @ "<<i<<" = "<<aMom[i]<<" nMom @ "<<i<<" = "<<nMom[i]<<std::endl;
    }

    return 0;
}
