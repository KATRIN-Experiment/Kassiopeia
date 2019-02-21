#include <cstdlib>
#include <iostream>
#include <cmath>

#include "KThreeVector_KEMField.hh"
#include "KVMPathIntegral.hh"
#include "KVMLineIntegral.hh"
#include "KVMSurfaceIntegral.hh"
#include "KVMFluxIntegral.hh"

#include "KVMField.hh"
#include "KVMFieldWrapper.hh"

#include "KVMLineSegment.hh"
#include "KVMTriangularSurface.hh"
#include "KVMRectangularSurface.hh"



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

        void SetOrigin(KThreeVector origin){fOrigin = origin;};

    private:

        KThreeVector fOrigin;

};

int main(int /*argc*/, char** /*argv[]*/)
{

    //geometry for the electrodes
    double z_offset = 0.0;

    //triangle descriptors
    KThreeVector v1(1.,0.,0.);
    KThreeVector v2(-0.5, std::sqrt(3.)/2.0, z_offset);
    KThreeVector v3(-0.5, -1.0*std::sqrt(3.0)/2.0, z_offset);
    KThreeVector axis1 = (v2 - v1).Unit();
    KThreeVector axis2 = (v3 - v1).Unit();
    axis1 = axis1.Unit();
    axis2 = axis2.Unit();
    double Tsize = 1;
    double TriN1[3] = {axis1[0], axis1[1], axis1[2]}; //direction of side 1
    double TriN2[3] = {axis2[0], axis2[1], axis2[2]}; //direction of side 2
    double TriA = Tsize*(v2 - v1).Magnitude();//1.732038; //length of side 1
    double TriB = Tsize*(v3 - v1).Magnitude();//1.732038; //lenght of side 2
    double TriP[3] = {Tsize,0,z_offset}; //corner1
    KThreeVector TriP2;//corner2
    KThreeVector TriP3; //corner3
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
    //double WireDiameter = 0.01;
    double WireStartPoint[3] = {0,0,0};
    KThreeVector dir(1.,2.,3.); dir = dir.Unit();
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

    //first we create and instance of the class which has the function(s) we want
    //to integrate
    FieldTestCase testcase;

    //set some internal variables
    KThreeVector o(0.,0.,0.);
    testcase.SetOrigin(o);

    //all functions that we wish to integrate over must take as an argument
    //a 3-element array (some point in space). They can return any number of variables
    //but this number must be specified.

    //all member functions that are integrated over must follow the signature:
    //void FunctionName(const double* input, double* output) const;
    //below are some examples

    //here we wrap a function which takes a point in 3-space and returns 1 value
//    typedef KVMFieldWrapper<FieldTestCase, &FieldTestCase::identity> identity_func;
//    identity_func* ident = new identity_func(&testcase,3,1);

    typedef KVMFieldWrapper<FieldTestCase, &FieldTestCase::dist> dist_func;
    dist_func* dist = new dist_func(&testcase,3,1);

    //here we wrap a function which takes a point in 3-space and returns 6 values
//    typedef KVMFieldWrapper<FieldTestCase, &FieldTestCase::functionA> funcA;
//    funcA* funcATest = new funcA(&testcase,3,6);

    //here we wrap a function which takes a point in 3-space and returns 3 values
//    typedef KVMFieldWrapper<FieldTestCase, &FieldTestCase::radius_vec> radVec;
//    radVec* radVecTest = new radVec(&testcase,3,3);

    std::cout.precision(16);

    //make sure to allocate enough space to retrieve results
    double result[6];

    //create a new surface integrator
    KVMSurfaceIntegral<1>* surfaceInt1D = new KVMSurfaceIntegral<1>(); //integrates a 1-dim function
    KVMSurfaceIntegral<3>* surfaceInt3D = new KVMSurfaceIntegral<3>(); //integrates a 3-dim function
    KVMSurfaceIntegral<6>* surfaceInt6D = new KVMSurfaceIntegral<6>(); //integrates a 6-dim function

    //make sure we intialize the integrators, lets use 8-th order quadrature for all
    surfaceInt1D->SetNTerms(8);
    surfaceInt3D->SetNTerms(8);
    surfaceInt6D->SetNTerms(8);

    //set the surface, in this case we pass a triangle electrode
    surfaceInt1D->SetSurface(triangle);
    //set the (wrapped) function we want to integrate
    surfaceInt1D->SetField(dist);
    //perform the integration
    surfaceInt1D->Integral(result);
    std::cout<<"Integral over triangle : "<<result[0]<<std::endl;

//    //now lets integrate a different function,
//    //but with a pointer to some unknown electrode whose type we are not certain of
//    //(here we use a rectangle)
//    const KTElectrode* anon;
//    anon = rectangle;
//    //if the type of electrode is not recognized by the integrator then 'SetElectrode' will
//    //return false, the only recognized types are wire, triangle, and rectangle

//    bool success = elecInt->SetElectrode(anon);
//    if(success) //go ahead, set function and integrate
//    {
//        elecInt->SetField(funcATest);
//        elecInt->Integral(result);
//        std::cout<<"Integral over triangle electrode: "<<std::endl;
//        for(unsigned int i=0; i<6; i++)
//        {
//            std::cout<<"result["<<i<<"] = "<<result[i]<<std::endl;
//        }
//    }

//    //now lets do the same with a wire
//    anon = wire;
//    success = elecInt->SetElectrode(anon);
//    if(success) //go ahead, set function and integrate
//    {
//        //if we are using the  same function there is no need to do this twice, but do it anyway
//        elecInt->SetField(funcATest);
//        elecInt->Integral(result);
//        std::cout<<"Integral over wire electrode: "<<std::endl;
//        for(unsigned int i=0; i<6; i++)
//        {
//            std::cout<<"result["<<i<<"] = "<<result[i]<<std::endl;
//        }
//    }


//    //clean up
//    delete triangle;
//    delete rectangle;
//    delete wire;
//    delete elecInt;
//    delete funcATest;
//    delete ident;
//    delete radVecTest;

    return 0;
}
