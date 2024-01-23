/*
* KElectrostaticImportField.cc
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#include "KElectrostaticImportField.hh"
#include "KEMFileInterface.hh"
#include <iostream>
#include <fstream>
#include <string>

using namespace katrin;

namespace KEMField
{

KElectrostaticImportField::KElectrostaticImportField() :
    fFile("/home/wonyongc/sim/Eimports/CBF54_-0.2_T45_74366181.txt"),
    fSize(74366181),
    fXRange(KFieldVector {-0.053,0.053,0.0005}),
    fYRange(KFieldVector {-0.05, 0.05, 0.0005}),
    fZRange(KFieldVector {-0.4, 0.468, 0.0005}),
    nx(212),ny(200),nz(1736),
    xmin(-0.053),xmax(0.053),
    ymin(-0.05),ymax(0.05),
    zmin(-0.4),zmax(0.468)
{
    std::cout << "\nEntered KElectrostaticImportField constructor\nfSize is: " << fSize<< " \n";

    fEx = new double[fSize];
    fEy = new double[fSize];
    fEz = new double[fSize];
    fPhi = new double[fSize];

//    auto* x = new double[fSize];
//    auto* y = new double[fSize];
//    auto* z = new double[fSize];

    std::cout << "Instantiated new Exyz double arrays \n";

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

    std::string exstr;
    std::string eystr;
    std::string ezstr;
    std::string phistr;

//    std::string xstr;
//    std::string ystr;
//    std::string zstr;

    std::ifstream filestream;
    filestream.open(fFile);

    std::string line;

    std::string::size_type sz;

    std::cout << "Opened E field file: " << fFile << "\n";

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

        getline(linestream, exstr, ',');
        getline(linestream, eystr, ',');
        getline(linestream, ezstr, ',');
        getline(linestream, phistr, ',');

        fEx[mi] = std::stod(exstr, &sz);
        fEy[mi] = std::stod(eystr, &sz);
        fEz[mi] = std::stod(ezstr, &sz);
        fPhi[mi] = std::stod(phistr, &sz);

        mi++;
    }
    filestream.close();
    std::cout << "Closed E field file \n";


    KFieldVector testpoint{0, 0, -0.1};
    KFieldVector testfield = ElectricFieldCore(testpoint);
    std::cout << "Test Electric Field at (0,0,-0.1): " << testfield[0] << " " << testfield[1] << " " << testfield[2]
              << "\n\n";

//    SaveFieldSamples();
}

void KElectrostaticImportField::SaveFieldSamples()
{
    std::ofstream test1, test2, test3, test4, test5, test6, test7, test8, test9;

    test1.open("/home/wonyongc/sim/fieldsamples/etest_x035_y-075.txt");
    test2.open("/home/wonyongc/sim/fieldsamples/etest_x035_y0.txt");
    test3.open("/home/wonyongc/sim/fieldsamples/etest_x035_y075.txt");

    test4.open("/home/wonyongc/sim/fieldsamples/etest_x0_y-075.txt");
    test5.open("/home/wonyongc/sim/fieldsamples/etest_x0_y0.txt");
    test6.open("/home/wonyongc/sim/fieldsamples/etest_x0_y075.txt");

    test7.open("/home/wonyongc/sim/fieldsamples/etest_x-035_y-075.txt");
    test8.open("/home/wonyongc/sim/fieldsamples/etest_x-035_y0.txt");
    test9.open("/home/wonyongc/sim/fieldsamples/etest_x-035_y075.txt");
    double xt;
    double yt;
    double zt;

    xt = 0.035;
    yt = -0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test1 << zt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test1.close();

    xt = 0.035;
    yt = 0;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test2 << xt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test2.close();

    xt = 0.035;
    yt = 0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test3 << yt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test3.close();

    xt = 0;
    yt = -0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test4 << xt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test4.close();

    xt = 0;
    yt = 0;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test5 << yt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test5.close();

    xt = 0;
    yt = 0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test6 << xt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test6.close();

    xt = -0.035;
    yt = -0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);

        test7 << yt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test7.close();

    xt = -0.035;
    yt = 0;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);
        test8 << zt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test8.close();

    xt = -0.035;
    yt = 0.0075;
    zt = -0.4;
    while (zt < 0.5)
    {
        KFieldVector testP {xt,yt,zt};
        KFieldVector testF = ElectricFieldCore(testP);
        double testPhi = PotentialCore(testP);
        test9 << zt << " " << testF[0] << " " << testF[1] << " " << testF[2] << " " << testPhi << "\n";
        zt += 0.0001;
    }
    test9.close();
}

KElectrostaticImportField::~KElectrostaticImportField() = default;

double KElectrostaticImportField::PotentialCore(const KPosition& P) const
{
    if ((P[0] > xmax || P[0] < xmin) ||
        (P[1] > ymax || P[1] < ymin) ||
        (P[2] > zmax || P[2] < zmin) )
    {
//        std::cout << "        Electric Potential out of bounds, returning -1 \n";
//        std::cout << "        x: " << P[0] << " y: " << P[1] << " z: " << P[2] << "\n";

        return -1000;
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
        std::cout << "Electric Potential c000: " << c000 << "\n";
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

    double phic00 = fPhi[c000]*(1-xd)+fPhi[c100]*xd;
    double phic01 = fPhi[c001]*(1-xd)+fPhi[c101]*xd;
    double phic10 = fPhi[c010]*(1-xd)+fPhi[c110]*xd;
    double phic11 = fPhi[c011]*(1-xd)+fPhi[c111]*xd;
    double phic0 = phic00*(1-yd)+phic10*yd;
    double phic1 = phic01*(1-yd)+phic11*yd;
    double phic = phic0*(1-zd)+phic1*zd;

    return phic;
}

KFieldVector KElectrostaticImportField::ElectricFieldCore(const KPosition& P) const {

    if ((P[0] > xmax || P[0] < xmin) ||
        (P[1] > ymax || P[1] < ymin) ||
        (P[2] > zmax || P[2] < zmin) )
    {
        double ex = 0;
        double ey = 0;
        double ez = 0;

        if (P[0] > xmax) ex = -10000;
        else if (P[0] < xmin) ex = 10000;
        if (P[1] > ymax) ey = -10000;
        else if (P[1] < ymin) ey = 10000;
        if (P[2] > zmax) ez = -10000;
        else if (P[2] < zmin) ez = 10000;

//        std::cout << "    Electric Field out of bounds, returning +/-1 \n";
//        std::cout << "    x: " << P[0] << " y: " << P[1] << " z: " << P[2] << "\n";

        return KFieldVector{ex,ey,ez};
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
        std::cout << "Electric Field c000: " << c000 << "\n";
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

    double exc00 = fEx[c000]*(1-xd)+fEx[c100]*xd;
    double exc01 = fEx[c001]*(1-xd)+fEx[c101]*xd;
    double exc10 = fEx[c010]*(1-xd)+fEx[c110]*xd;
    double exc11 = fEx[c011]*(1-xd)+fEx[c111]*xd;
    double exc0 = exc00*(1-yd)+exc10*yd;
    double exc1 = exc01*(1-yd)+exc11*yd;
    double exc = exc0*(1-zd)+exc1*zd;

    double eyc00 = fEy[c000]*(1-xd)+fEy[c100]*xd;
    double eyc01 = fEy[c001]*(1-xd)+fEy[c101]*xd;
    double eyc10 = fEy[c010]*(1-xd)+fEy[c110]*xd;
    double eyc11 = fEy[c011]*(1-xd)+fEy[c111]*xd;
    double eyc0 = eyc00*(1-yd)+eyc10*yd;
    double eyc1 = eyc01*(1-yd)+eyc11*yd;
    double eyc = eyc0*(1-zd)+eyc1*zd;

    double ezc00 = fEz[c000]*(1-xd)+fEz[c100]*xd;
    double ezc01 = fEz[c001]*(1-xd)+fEz[c101]*xd;
    double ezc10 = fEz[c010]*(1-xd)+fEz[c110]*xd;
    double ezc11 = fEz[c011]*(1-xd)+fEz[c111]*xd;
    double ezc0 = ezc00*(1-yd)+ezc10*yd;
    double ezc1 = ezc01*(1-yd)+ezc11*yd;
    double ezc = ezc0*(1-zd)+ezc1*zd;


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
    //                  << "fEx[c000]: " << fEx[c000] << "\n"
    //                  << "fEy[c000]: " << fEy[c000] << "\n"
    //                  << "fEz[c000]: " << fEz[c000] << "\n";
    //    }


    KFieldVector fval {exc, eyc, ezc};
    return fval;

}

void KElectrostaticImportField::SetXRange( const KFieldVector& aXRange ) {
    fXRange = aXRange;
    return;
}

void KElectrostaticImportField::SetYRange( const KFieldVector& aYRange ) {
    fYRange = aYRange;
    return;
}

void KElectrostaticImportField::SetZRange( const KFieldVector& aZRange ) {
    fZRange = aZRange;
    return;
}

} /* namespace KEMField */
