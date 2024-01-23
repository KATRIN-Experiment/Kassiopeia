/*
* KMagnetostaticImportField.cc
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#include "KMagnetostaticImportField.hh"
#include "KEMFileInterface.hh"
#include <iostream>
#include <fstream>
#include <string>

using namespace katrin;

namespace KEMField
{

KMagnetostaticImportField::KMagnetostaticImportField() :
    fFile("/home/wonyongc/sim/Bimports/ete_c5700_25n_tilt45_74366181.txt"),
    fSize(74366181),
    fXRange(KFieldVector {-0.053,0.053,0.0005}),
    fYRange(KFieldVector {-0.05, 0.05, 0.0005}),
    fZRange(KFieldVector {-0.4, 0.468, 0.0005}),
    nx(212),ny(200),nz(1736),
    xmin(-0.053),xmax(0.053),
    ymin(-0.05),ymax(0.05),
    zmin(-0.4),zmax(0.468)
{
    std::cout << "\nEntered KMagnetostaticImportField constructor\nfSize is: " << fSize << " \n";

    fBx = new double[fSize];
    fBy = new double[fSize];
    fBz = new double[fSize];

//    auto* x = new double[fSize];
//    auto* y = new double[fSize];
//    auto* z = new double[fSize];

    std::cout << "Instantiated new Bxyz double arrays \n";

    // nx = (int)((fXRange[1] - fXRange[0]) / fXRange[2]);
    // ny = (int)((fYRange[1] - fYRange[0]) / fYRange[2]);
    // nz = (int)((fZRange[1] - fZRange[0]) / fZRange[2]);

    std::cout << "Instantiated new nx,ny,nz ints " << nx << " " << ny << " " << nz << "\n";

//    int m = 0;
//
//    for (int i = 0; i < (nx + 1); i++) {
//        for (int j = 0; j < (ny + 1); j++) {
//            for (int k = 0; k < (nz + 1); k++) {
//
//                x[m] = fXRange[0] + i * fXRange[2];
//                y[m] = fYRange[0] + j * fYRange[2];
//                z[m] = fZRange[0] + k * fZRange[2];
//                m++;
//            }
//        }
//    }

    x1 = double{fXRange[0]};
    x2 = double{fXRange[1]};
    dx = double{fXRange[2]};

    y1 = double{fYRange[0]};
    y2 = double{fYRange[1]};
    dy = double{fYRange[2]};

    z1 = double{fZRange[0]};
    z2 = double{fZRange[1]};
    dz = double{fZRange[2]};

    std::cout << "Instantiated new doubles for ranges \n"
              << x1 << " " << x2 << " " << dx << "\n"
              << y1 << " " << y2 << " " << dy << "\n"
              << z1 << " " << z2 << " " << dz << "\n";

    std::string bxstr;
    std::string bystr;
    std::string bzstr;

//    std::string xstr;
//    std::string ystr;
//    std::string zstr;

    std::ifstream filestream;
    filestream.open(fFile);

    std::string line;

    std::string::size_type sz;

    std::cout << "Opened B field file: " << fFile << "\n";

    int mi = 0;

//    double xr;
//    double yr;
//    double zr;

    while (mi < fSize) {

        getline(filestream, line);
        std::stringstream linestream(line);

//        getline(linestream, xstr, ',');
//        getline(linestream, ystr, ',');
//        getline(linestream, zstr, ',');

//        xr = std::stod(xstr, &sz);
//        yr = std::stod(ystr, &sz);
//        zr = std::stod(zstr, &sz);

//        if (round(x[mi] * 100000) != round(xr * 100000))
//            std::cout << "x mismatch: x[" << mi << "]: " << x[mi] << " xr[" << mi << "]: " << xr << "\n";
//        if (round(y[mi] * 100000) != round(yr * 100000))
//            std::cout << "y mismatch: y[" << mi << "]: " << y[mi] << " yr[" << mi << "]: " << yr << "\n";
//        if (round(z[mi] * 100000) != round(zr * 100000))
//            std::cout << "z mismatch: z[" << mi << "]: " << z[mi] << " zr[" << mi << "]: " << zr << "\n";

        getline(linestream, bxstr, ',');
        getline(linestream, bystr, ',');
        getline(linestream, bzstr, ',');


        fBx[mi] = std::stod(bxstr, &sz);
        fBy[mi] = std::stod(bystr, &sz);
        fBz[mi] = std::stod(bzstr, &sz);

        mi++;
    }
    filestream.close();
    std::cout << "Closed B field file \n";


    KFieldVector testpoint{0, 0, -0.1};
    KFieldVector testfield = MagneticFieldCore(testpoint);
    std::cout << "Test Magnetic Field at (0,0,-0.1): " << testfield[0] << " " << testfield[1] << " " << testfield[2]
              << "\n\n";

//    SaveFieldSamples();
}

void KMagnetostaticImportField::SaveFieldSamples()
{
    std::ofstream test1, test2, test3, test4, test5, test6, test7, test8, test9;

    test1.open("/home/wonyongc/sim/fieldsamples/btest_x035_y-075.txt");
    test2.open("/home/wonyongc/sim/fieldsamples/btest_x035_y0.txt");
    test3.open("/home/wonyongc/sim/fieldsamples/btest_x035_y075.txt");

    test4.open("/home/wonyongc/sim/fieldsamples/btest_x0_y-075.txt");
    test5.open("/home/wonyongc/sim/fieldsamples/btest_x0_y0.txt");
    test6.open("/home/wonyongc/sim/fieldsamples/btest_x0_y075.txt");

    test7.open("/home/wonyongc/sim/fieldsamples/btest_x-035_y-075.txt");
    test8.open("/home/wonyongc/sim/fieldsamples/btest_x-035_y0.txt");
    test9.open("/home/wonyongc/sim/fieldsamples/btest_x-035_y075.txt");

    double xt;
    double yt;
    double zt;

    xt = 0.035;
    yt = -0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test1 << zt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test1.close();

    xt = 0.035;
    yt = 0;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test2 << xt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test2.close();

    xt = 0.035;
    yt = 0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test3 << yt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test3.close();

    xt = 0;
    yt = -0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test4 << xt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test4.close();

    xt = 0;
    yt = 0;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test5 << yt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test5.close();

    xt = 0;
    yt = 0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test6 << xt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test6.close();

    xt = -0.035;
    yt = -0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test7 << yt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test7.close();

    xt = -0.035;
    yt = 0;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test8 << zt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test8.close();

    xt = -0.035;
    yt = 0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = MagneticFieldCore(testP);
        test9 << zt << " " << testF[0] << " " << testF[1] << " " << testF[2] << "\n";
        zt += 0.0001;
    }
    test9.close();

}

KMagnetostaticImportField::~KMagnetostaticImportField() = default;

KFieldVector KMagnetostaticImportField::MagneticPotentialCore(const KPosition& /*aSamplePoint*/) const
{
    KFieldVector aPotential {0,0,0};
    return aPotential;
}

KFieldVector KMagnetostaticImportField::MagneticFieldCore(const KPosition& P) const
{
    if ((P[0] > xmax || P[0] < xmin) ||
        (P[1] > ymax || P[1] < ymin) ||
        (P[2] > zmax || P[2] < zmin) )
    {
        double bx = -100;
        double by = 0;
        double bz = 0;
//
//        if (P[0] > xmax) bz = -0.0001;
//        else if (P[0] < xmin) bx = 1;
        if (P[1] > ymax) bx = -100;
//        else if (P[1] < ymin) by = 0;
//        if (P[2] > zmax) bz = 0;
//        else if (P[2] < zmin) bz = 0;

//        std::cout << "Magnetic Field out of bounds, returning const 1T field \n";
//        std::cout << "x: " << P[0] << " y: " << P[1] << " z: " << P[2] << "\n";

        return KFieldVector{bx,by,bz};
    }

    int ix = (int)(std::abs(P[0]-x1)/(dx));
    int iy = (int)(std::abs(P[1]-y1)/(dy));
    int iz = (int)(std::abs(P[2]-z1)/(dz));

    int c000 = ix*((nz+1)*(ny+1))+iy*(nz+1)+iz;

    int c100 = (ix+1)*((nz+1)*(ny+1))+iy*(nz+1)+iz;
    int c010 = ix*((nz+1)*(ny+1))+(iy+1)*(nz+1)+iz;
    int c001 = ix*((nz+1)*(ny+1))+iy*(nz+1)+(iz+1);

    int c110 = (ix+1)*((nz+1)*(ny+1))+(iy+1)*(nz+1)+iz;
    int c101 = (ix+1)*((nz+1)*(ny+1))+iy*(nz+1)+(iz+1);
    int c011 = ix*((nz+1)*(ny+1))+(iy+1)*(nz+1)+(iz+1);

    int c111 = (ix+1)*((nz+1)*(ny+1))+(iy+1)*(nz+1)+(iz+1);

    if (c000 > fSize-1) {
        std::cout << "Magnetic Field c000: " << c000 << "\n";
        std::cout << "x: " << P[0] << " y: " << P[1] << " z: " << P[2] << "\n";
        std::cout << "ix: " << ix << " iy: " << iy << " iz: " << iz << "\n";
    }
//    if (c100 > fSize-1) std::cout << "c100: " << c100 << "\n";
//    if (c010 > fSize-1) std::cout << "c010: " << c010 << "\n";
//    if (c001 > fSize-1) std::cout << "c001: " << c001 << "\n";
//    if (c110 > fSize-1) std::cout << "c110: " << c110 << "\n";
//    if (c101 > fSize-1) std::cout << "c101: " << c101 << "\n";
//    if (c011 > fSize-1) std::cout << "c011: " << c011 << "\n";
//    if (c111 > fSize-1) std::cout << "c111: " << c111 << "\n";


    double xd = (std::abs(P[0]-x1) -dx*ix ) /(dx);
    double yd = (std::abs(P[1]-y1) -dy*iy ) /(dy);
    double zd = (std::abs(P[2]-z1) -dz*iz ) /(dz);

    double bxc00 = fBx[c000]*(1-xd)+fBx[c100]*xd;
    double bxc01 = fBx[c001]*(1-xd)+fBx[c101]*xd;
    double bxc10 = fBx[c010]*(1-xd)+fBx[c110]*xd;
    double bxc11 = fBx[c011]*(1-xd)+fBx[c111]*xd;
    double bxc0 = bxc00*(1-yd)+bxc10*yd;
    double bxc1 = bxc01*(1-yd)+bxc11*yd;
    double bxc = bxc0*(1-zd)+bxc1*zd;

    double byc00 = fBy[c000]*(1-xd)+fBy[c100]*xd;
    double byc01 = fBy[c001]*(1-xd)+fBy[c101]*xd;
    double byc10 = fBy[c010]*(1-xd)+fBy[c110]*xd;
    double byc11 = fBy[c011]*(1-xd)+fBy[c111]*xd;
    double byc0 = byc00*(1-yd)+byc10*yd;
    double byc1 = byc01*(1-yd)+byc11*yd;
    double byc = byc0*(1-zd)+byc1*zd;

    double bzc00 = fBz[c000]*(1-xd)+fBz[c100]*xd;
    double bzc01 = fBz[c001]*(1-xd)+fBz[c101]*xd;
    double bzc10 = fBz[c010]*(1-xd)+fBz[c110]*xd;
    double bzc11 = fBz[c011]*(1-xd)+fBz[c111]*xd;
    double bzc0 = bzc00*(1-yd)+bzc10*yd;
    double bzc1 = bzc01*(1-yd)+bzc11*yd;
    double bzc = bzc0*(1-zd)+bzc1*zd;


//    if (debug) {
//        std::cout << "ix: " << ix << "\n"
//                  << "iy: " << iy << "\n"
//                  << "iz: " << iz << "\n"
//                  << "c000: " << c000 << "\n"
//                  << "c100: " << c100 << "\n"
//                  << "c010: " << c010 << "\n"
//                  << "c001: " << c001 << "\n"
//                  << "c110: " << c110 << "\n"
//                  << "c101: " << c101 << "\n"
//                  << "c011: " << c011 << "\n"
//                  << "c111: " << c111 << "\n"
//                  << "xd: " << xd << "\n"
//                  << "yd: " << yd << "\n"
//                  << "zd: " << zd << "\n"
//                  << "fBx[c000]: " << fBx[c000] << "\n"
//                  << "fBy[c000]: " << fBy[c000] << "\n"
//                  << "fBz[c000]: " << fBz[c000] << "\n";
//    }


    KFieldVector fval {bxc, byc, bzc};
    return fval;

}

KGradient KMagnetostaticImportField::MagneticGradientCore(const KPosition& /*aSamplePoint*/) const
{
    KGradient aGradient;
    return aGradient;
}

void KMagnetostaticImportField::SetXRange( const KFieldVector& aXRange ) {
    fXRange = aXRange;
    return;
}

void KMagnetostaticImportField::SetYRange( const KFieldVector& aYRange ) {
    fYRange = aYRange;
    return;
}

void KMagnetostaticImportField::SetZRange( const KFieldVector& aZRange ) {
    fZRange = aZRange;
    return;
}

} /* namespace KEMField */
