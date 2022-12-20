#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sstream>

#ifdef KEMFIELD_USE_GSL
#include <gsl/gsl_rng.h>
#endif


#include "KFMElectrostaticMultipoleCalculatorAnalytic.hh"
#include "KFMElectrostaticMultipoleCalculatorNumeric.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMPoint.hh"
#include "KFMPointCloud.hh"
#include "KThreeVector_KEMField.hh"
#include "KVMField.hh"
#include "KVMFieldWrapper.hh"
#include "KVMFluxIntegral.hh"
#include "KVMLineIntegral.hh"
#include "KVMLineSegment.hh"
#include "KVMPathIntegral.hh"
#include "KVMRectangularSurface.hh"
#include "KVMSurfaceIntegral.hh"
#include "KVMTriangularSurface.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticMultipoleCalculator_OpenCL.hh"
#endif

using namespace KEMField;
using katrin::KThreeVector;


#ifdef KEMFIELD_USE_GSL
//generates a random acute triangle with vertices on the sphere of radius r
void GenerateTriangle(gsl_rng* rng, double r, double& a, double& b, double* p0, double* n1, double* n2)
{
    double x = 0;
    double y = 0;
    double z = 0;
    double norm = 0;
    double p1[3];
    double p2[3];

    //first generate the point p0 within the sphere
    //generate the points on the sphere surface
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    p0[0] = r * (x / norm);
    p0[1] = r * (y / norm);
    p0[2] = r * (z / norm);

    //now generate the second point
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    p1[0] = r * (x / norm);
    p1[1] = r * (y / norm);
    p1[2] = r * (z / norm);

    //now generate the third point
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    p2[0] = r * (x / norm);
    p2[1] = r * (y / norm);
    p2[2] = r * (z / norm);


    //now make sure that it is an acute triangle

    KThreeVector v0(p0[0], p0[1], p0[2]);
    KThreeVector v1(p1[0], p1[1], p1[2]);
    KThreeVector v2(p2[0], p2[1], p2[2]);
    KThreeVector vn1 = (v1 - v0);
    a = vn1.Magnitude();
    vn1 = vn1.Unit();
    n1[0] = vn1.X();
    n1[1] = vn1.Y();
    n1[2] = vn1.Z();

    KThreeVector vn2 = (v2 - v0);
    b = vn2.Magnitude();
    vn2 = vn2.Unit();
    n2[0] = vn2.X();
    n2[1] = vn2.Y();
    n2[2] = vn2.Z();
}


//generates a random acute triangle with vertices on the sphere of radius r
void GenerateRectangle(gsl_rng* rng, double r, double& a, double& b, double* p0, double* n1, double* n2)
{
    double x = 0;
    double y = 0;
    double z = 0;
    double norm = 0;
    double n4[4];

    //first generate the point p0 within the sphere
    //generate the points on the sphere surface
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    p0[0] = x;
    p0[1] = y;
    p0[2] = z;

    //now generate a random direction
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    n1[0] = r * (x / norm);
    n1[1] = r * (y / norm);
    n1[2] = r * (z / norm);

    //now generate a second random direction
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    n4[0] = r * (x / norm);
    n4[1] = r * (y / norm);
    n4[2] = r * (z / norm);

    //take cross product with first direction so we know we have perpendicular vectors
    KThreeVector v1(n1[0], n1[1], n1[2]);
    KThreeVector v4(n4[0], n4[1], n4[2]);
    KThreeVector v2 = v1.Cross(v4);
    v2 = v2.Unit();
    n2[0] = v2[0];
    n2[1] = v2[1];
    n2[2] = v2[2];

    //generate two random lengths
    a = gsl_rng_uniform(rng) * (2.0 * r);
    b = gsl_rng_uniform(rng) * (2.0 * r);
}

//generates a random acute triangle with vertices on the sphere of radius r
void GenerateWire(gsl_rng* rng, double r, double* p0, double* p1)
{
    double x = 0;
    double y = 0;
    double z = 0;
    double norm = 0;

    //first generate the point p0 within the sphere
    //generate the points on the sphere surface
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    p0[0] = r * (x / norm);
    p0[1] = r * (y / norm);
    p0[2] = r * (z / norm);

    //now generate the second point
    do {
        x = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        y = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        z = -1.0 * r + gsl_rng_uniform(rng) * (2.0 * r);
        norm = std::sqrt(x * x + y * y + z * z);
    } while (norm > r);
    p1[0] = r * (x / norm);
    p1[1] = r * (y / norm);
    p1[2] = r * (z / norm);
}

#endif


int main(int argc, char* argv[])
{

    std::string usage =
        "\n"
        "Usage: TestMultipoleCalculators <options>\n"
        "\n"
        "This program computes the multipole coefficients for BEM elements with various calculator types. \n"
        "\tAvailable options:\n"
        "\t -h, --help               (shows this message and exits)\n"
        "\t -t  --type               (triangle=0, rectangle=1, wire=2)\n"
        "\t -m, --mode               (0: fixed data, or 1: random)\n"
        "\t -s, --samples            (if in random mode, number of samples to use)\n"
        "\t -o, --ocl-force          (0: OpenCL default, 1: force OpenCL analytic, 2: force OpenCL numerical) \n"
        "\t -d, --degree             (degree of multipole expansions) \n";

    std::string config_file;
    unsigned int type_select = 0;
    unsigned int mode = 0;
    unsigned int samples = 10;
    (void) samples;  //remove unused var warning when GSL is not available
    unsigned int oclforce = 0;
    (void) oclforce;
    unsigned int degree = 3;

    static struct option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                          {"type", required_argument, nullptr, 't'},
                                          {"mode", required_argument, nullptr, 'm'},
                                          {"samples", required_argument, nullptr, 's'},
                                          {"ocl-force", required_argument, nullptr, 'o'},
                                          {"degree", required_argument, nullptr, 'd'}};

    static const char* optString = "ht:m:s:o:d:";

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('t'):
                type_select = atoi(optarg);
                break;
            case ('m'):
                mode = atoi(optarg);
                break;
            case ('s'):
                samples = atoi(optarg);
                break;
            case ('o'):
                oclforce = atoi(optarg);
                break;
            case ('d'):
                degree = atoi(optarg);
                break;
            default:
                std::cout << usage << std::endl;
                return 1;
        }
    }

    unsigned int nquad = 4;
    if (degree > nquad) {
        nquad = degree;
    };


    KFMElectrostaticParameters params;
    params.degree = degree;
    params.verbosity = 3;

    //now lets make the multipole calculators
    auto* aCalc = new KFMElectrostaticMultipoleCalculatorAnalytic();
    aCalc->SetDegree(degree);

    auto* nCalc = new KFMElectrostaticMultipoleCalculatorNumeric();
    nCalc->SetDegree(degree);
    nCalc->SetNumberOfQuadratureTerms(nquad);

#ifdef KEMFIELD_USE_OPENCL
    bool standalone = true;
    KFMElectrostaticMultipoleCalculator_OpenCL* oclCalc = new KFMElectrostaticMultipoleCalculator_OpenCL(standalone);
    oclCalc->SetParameters(params);
    oclCalc->UseDefault();
    if (oclforce == 1) {
        oclCalc->ForceAnalytic();
    };
    if (oclforce == 2) {
        oclCalc->ForceNumerical();
    };
    oclCalc->Initialize();
#endif


    if (mode == 0)  //fixed geometry
    {

        //geometry for the electrodes
        double z_offset = 1.1;

        //triangle descriptors
        KThreeVector v1(-1., 0., 0.);
        KThreeVector v2(0.5, std::sqrt(3.) / 2.0, z_offset);
        KThreeVector v3(0.5, -1.0 * std::sqrt(3.0) / 2.0, z_offset);

        KThreeVector axis1 = (v2 - v1).Unit();
        KThreeVector axis2 = (v3 - v1).Unit();
        axis1 = axis1.Unit();
        axis2 = axis2.Unit();
        double Tsize = 1;
        double TriN1[3] = {axis1[0], axis1[1], axis1[2]};  //direction of side 1
        double TriN2[3] = {axis2[0], axis2[1], axis2[2]};  //direction of side 2
        double TriA = Tsize * (v2 - v1).Magnitude();       //1.732038; //length of side 1
        double TriB = Tsize * (v3 - v1).Magnitude();       //1.732038; //lenght of side 2
        double TriP[3] = {0.0, 0.0, 0.};                   //{v1.X(), v1.Y(), v1.Z()};// {Tsize,0,z_offset}; //corner1
        KThreeVector TriP2;                                //corner2
        KThreeVector TriP3;                                //corner3
        TriP2.SetComponents(TriP[0], TriP[1], TriP[2]);
        TriP2 += TriA * axis1;
        TriP3.SetComponents(TriP[0], TriP[1], TriP[2]);
        TriP3 += TriB * axis2;

        //geometry descriptors for the rectangle used
        //sides parallel to x and y axes respectively

        double RecA = 1.1010101;        //length of side 1
        double RecB = 29.0;             //length of side 2
        double RecP[3] = {0, 0, 0};     //corner
        double RecN1[3] = {1.0, 0, 0};  //direction of side 1
        double RecN2[3] = {0, 1.0, 0};  //direction of side 2

        //geometry for the wire electrode used
        double WireLength = 1.1010101;
        //    double WireDiameter = 0.01;
        double WireStartPoint[3] = {0, 0, 0};
        KThreeVector dir(1., 2., 3.);
        dir = dir.Unit();
        double WireDirection[3] = {dir[0], dir[1], dir[2]};
        double WireEndPoint[3] = {WireStartPoint[0] + WireLength * WireDirection[0],
                                  WireStartPoint[1] + WireLength * WireDirection[1],
                                  WireStartPoint[2] + WireLength * WireDirection[2]};

        //set up line segment, and triangle, rectangle surfaces
        auto* line = new KVMLineSegment();
        line->SetAll(WireStartPoint, WireEndPoint);
        line->Initialize();

        auto* triangle = new KVMTriangularSurface();
        triangle->SetAll(TriP, TriN1, TriN2, TriA, TriB);
        triangle->Initialize();

        auto* rectangle = new KVMRectangularSurface();
        rectangle->SetAll(RecP, RecN1, RecN2, RecA, RecB);
        rectangle->Initialize();

        //extract their vertices
        std::vector<KFMPoint<3>> triangleVertices;
        triangleVertices.resize(3);
        std::vector<KFMPoint<3>> rectangleVertices;
        rectangleVertices.resize(4);
        std::vector<KFMPoint<3>> wireVertices;
        wireVertices.resize(2);

        triangle->GetVertices((triangleVertices[0]), (triangleVertices[1]), (triangleVertices[2]));
        rectangle->GetVertices((rectangleVertices[0]),
                               (rectangleVertices[1]),
                               (rectangleVertices[2]),
                               (rectangleVertices[3]));
        line->GetVertices((wireVertices[0]), (wireVertices[1]));

        //insert into point clouds
        KFMPointCloud<3> triangleCloud;
        triangleCloud.SetPoints(&triangleVertices);
        KFMPointCloud<3> rectangleCloud;
        rectangleCloud.SetPoints(&rectangleVertices);
        KFMPointCloud<3> wireCloud;
        wireCloud.SetPoints(&wireVertices);


        double origin[3];
        KThreeVector origin_vec(0., 0., -1.0);
        origin_vec = origin_vec.Unit();
        double mag = 1.0;
        origin_vec *= mag;

        origin[0] = 0.0;
        origin[1] = 1e-5;
        origin[2] = 1.0 - 1e-5;

        KFMScalarMultipoleExpansion aExpan;
        KFMScalarMultipoleExpansion nExpan;
        KFMScalarMultipoleExpansion oclExpan;
        aExpan.SetDegree(degree);
        nExpan.SetDegree(degree);
        oclExpan.SetDegree(degree);

        switch (type_select) {
            case 0:
                //aCalc->ConstructExpansion(origin, &triangleCloud, &aExpan);
                aCalc->ConstructExpansion(origin, &triangleCloud, &aExpan);
                nCalc->ConstructExpansion(origin, &triangleCloud, &nExpan);
#ifdef KEMFIELD_USE_OPENCL
                oclCalc->ConstructExpansion(origin, &triangleCloud, &oclExpan);
#endif
                break;
            case 1:
                aCalc->ConstructExpansion(origin, &rectangleCloud, &aExpan);
                nCalc->ConstructExpansion(origin, &rectangleCloud, &nExpan);
#ifdef KEMFIELD_USE_OPENCL
                oclCalc->ConstructExpansion(origin, &rectangleCloud, &oclExpan);
#endif
                break;
            case 2:
                aCalc->ConstructExpansion(origin, &wireCloud, &aExpan);
                nCalc->ConstructExpansion(origin, &wireCloud, &nExpan);
#ifdef KEMFIELD_USE_OPENCL
                oclCalc->ConstructExpansion(origin, &wireCloud, &oclExpan);
#endif
                break;
        }

        std::vector<std::complex<double>> aMom;
        std::vector<std::complex<double>> nMom;
        std::vector<std::complex<double>> oclMom;

        aExpan.GetMoments(&aMom);
        nExpan.GetMoments(&nMom);
        oclExpan.GetMoments(&oclMom);

        for (unsigned int i = 0; i < aMom.size(); i++) {
            std::cout << "(analytic vs numerical) difference @ " << i << " = " << std::abs(aMom[i] - nMom[i])
                      << std::endl;
        }

        for (unsigned int i = 0; i < aMom.size(); i++) {
            std::cout << "aMom @ " << i << " = " << aMom[i] << " nMom @ " << i << " = " << nMom[i] << std::endl;
        }

#ifdef KEMFIELD_USE_OPENCL
        for (unsigned int i = 0; i < aMom.size(); i++) {
            std::cout << "(analytic vs OpenCL)  difference @ " << i << " = " << std::abs(aMom[i] - oclMom[i])
                      << std::endl;
        }

        for (unsigned int i = 0; i < aMom.size(); i++) {
            std::cout << "aMom @ " << i << " = " << aMom[i] << " oclMom @ " << i << " = " << oclMom[i] << std::endl;
        }
#endif
    }
    else if (mode == 1)  //sample random shapes and compute average errors
    {
#ifdef KEMFIELD_USE_GSL

        double origin[3];
        origin[0] = 0.0;
        origin[1] = 1e-5;
        origin[2] = 1.0 - 1e-5;

        const gsl_rng_type* T;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        gsl_rng* rng = gsl_rng_alloc(T);

        double radius = 2;

        KFMScalarMultipoleExpansion aExpan;
        KFMScalarMultipoleExpansion nExpan;
        KFMScalarMultipoleExpansion oclExpan;
        aExpan.SetDegree(degree);
        nExpan.SetDegree(degree);
        oclExpan.SetDegree(degree);


        double worst_absolute_error_numeric = 0;
        double worst_absolute_error_ocl = 0;

        for (unsigned int s = 0; s < samples; s++) {

            switch (type_select) {
                case 0: {
                    //triangle data
                    double TriN1[3];
                    double TriN2[3];
                    double TriA;
                    double TriB;
                    double TriP[3];
                    GenerateTriangle(rng, radius, TriA, TriB, TriP, TriN1, TriN2);
                    KVMTriangularSurface* triangle = new KVMTriangularSurface();
                    triangle->SetAll(TriP, TriN1, TriN2, TriA, TriB);
                    triangle->Initialize();
                    std::vector<KFMPoint<3>> triangleVertices;
                    triangleVertices.resize(3);
                    triangle->GetVertices((triangleVertices[0]), (triangleVertices[1]), (triangleVertices[2]));
                    KFMPointCloud<3> triangleCloud;
                    triangleCloud.SetPoints(&triangleVertices);
                    //aCalc->ConstructExpansion(origin, &triangleCloud, &aExpan);
                    aCalc->ConstructExpansion(origin, &triangleCloud, &aExpan);
                    nCalc->ConstructExpansion(origin, &triangleCloud, &nExpan);
#ifdef KEMFIELD_USE_OPENCL
                    oclCalc->ConstructExpansion(origin, &triangleCloud, &oclExpan);
#endif
                } break;
                case 1: {
                    //rectangle data
                    double RecA;
                    double RecB;
                    double RecP[3];
                    double RecN1[3];
                    double RecN2[3];
                    GenerateRectangle(rng, radius, RecA, RecB, RecP, RecN1, RecN2);
                    KVMRectangularSurface* rectangle = new KVMRectangularSurface();
                    rectangle->SetAll(RecP, RecN1, RecN2, RecA, RecB);
                    rectangle->Initialize();
                    std::vector<KFMPoint<3>> rectangleVertices;
                    rectangleVertices.resize(4);
                    rectangle->GetVertices((rectangleVertices[0]),
                                           (rectangleVertices[1]),
                                           (rectangleVertices[2]),
                                           (rectangleVertices[3]));
                    KFMPointCloud<3> rectangleCloud;
                    rectangleCloud.SetPoints(&rectangleVertices);
                    aCalc->ConstructExpansion(origin, &rectangleCloud, &aExpan);
                    nCalc->ConstructExpansion(origin, &rectangleCloud, &nExpan);
#ifdef KEMFIELD_USE_OPENCL
                    oclCalc->ConstructExpansion(origin, &rectangleCloud, &oclExpan);
#endif
                } break;
                case 2: {
                    //wire data
                    double WireStartPoint[3];
                    double WireEndPoint[3];
                    GenerateWire(rng, radius, WireStartPoint, WireEndPoint);
                    KVMLineSegment* line = new KVMLineSegment();
                    line->SetAll(WireStartPoint, WireEndPoint);
                    line->Initialize();
                    std::vector<KFMPoint<3>> wireVertices;
                    wireVertices.resize(2);
                    line->GetVertices((wireVertices[0]), (wireVertices[1]));
                    KFMPointCloud<3> wireCloud;
                    wireCloud.SetPoints(&wireVertices);
                    aCalc->ConstructExpansion(origin, &wireCloud, &aExpan);
                    nCalc->ConstructExpansion(origin, &wireCloud, &nExpan);
#ifdef KEMFIELD_USE_OPENCL
                    oclCalc->ConstructExpansion(origin, &wireCloud, &oclExpan);
#endif
                } break;
            }

            std::vector<std::complex<double>> aMom;
            std::vector<std::complex<double>> nMom;
            std::vector<std::complex<double>> oclMom;

            aExpan.GetMoments(&aMom);
            nExpan.GetMoments(&nMom);
            oclExpan.GetMoments(&oclMom);

            for (unsigned int i = 0; i < aMom.size(); i++) {
                if (std::abs(aMom[i] - nMom[i]) > worst_absolute_error_numeric) {
                    worst_absolute_error_numeric = std::abs(aMom[i] - nMom[i]);
                }

                if (std::abs(aMom[i] - oclMom[i]) > worst_absolute_error_ocl) {
                    worst_absolute_error_ocl = std::abs(aMom[i] - oclMom[i]);
                }
            }
        }

        std::cout << "worst error numerical integrator w.r.t analytic: " << worst_absolute_error_numeric << std::endl;
        std::cout << "worst error opencl integrator w.r.t analytic: " << worst_absolute_error_ocl << std::endl;

#endif
    }


    return 0;
}
