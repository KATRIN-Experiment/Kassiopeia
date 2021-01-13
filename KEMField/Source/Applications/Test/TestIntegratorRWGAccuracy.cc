#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KThreeVector_KEMField.hh"

#include <cstdlib>
#include <ctime>
#include <getopt.h>
#include <iomanip>
#include <iostream>

using namespace KEMField;

typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle> KEMTriangle;
using KEMRectangle = KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>;

double IJKLRANDOM;
void subrn(double* u, int len);
double randomnumber();

#define ELEMENTNO 10

double fToleranceLambda = 1.E-15; /* tolerance for determining if field point is on vertex */

void AddTriangle(std::vector<KEMTriangle>& v, KFieldVector triP0, KFieldVector triP1, KFieldVector triP2)
{
    KEMTriangle newTri;
    newTri.SetA(sqrt(POW2(triP1[0] - triP0[0]) + POW2(triP1[1] - triP0[1]) + POW2(triP1[2] - triP0[2])));
    newTri.SetB(sqrt(POW2(triP2[0] - triP0[0]) + POW2(triP2[1] - triP0[1]) + POW2(triP2[2] - triP0[2])));
    newTri.SetP0(triP0);

    newTri.SetN1(KFieldVector((triP1[0] - triP0[0]) / newTri.GetA(),
                              (triP1[1] - triP0[1]) / newTri.GetA(),
                              (triP1[2] - triP0[2]) / newTri.GetA()));

    newTri.SetN2(KFieldVector((triP2[0] - triP0[0]) / newTri.GetB(),
                              (triP2[1] - triP0[1]) / newTri.GetB(),
                              (triP2[2] - triP0[2]) / newTri.GetB()));

    v.push_back(newTri);
}

int main()
{
    bool print = true;

    // Boundary integrators
    KElectrostaticBiQuadratureTriangleIntegrator intTriQuad;
    KElectrostaticAnalyticTriangleIntegrator intTriAna;
    KElectrostaticRWGTriangleIntegrator intTriRwg;

    KElectrostaticBiQuadratureRectangleIntegrator intReQuad;
    KElectrostaticAnalyticRectangleIntegrator intReAna;
    KElectrostaticRWGRectangleIntegrator intReRwg;

    const unsigned int elementNo(ELEMENTNO);

    for (unsigned int i = 0; i < elementNo; i++) {
        ///////////////////
        // DICE TRIANGLE //
        ///////////////////

        double triP0[3];
        double triP1[3];
        double triP2[3];

        // dice triangle geometry

        IJKLRANDOM = i + 1;

        if (print)
            std::cout << "P0 = ";
        for (double& l : triP0) {
            l = -1. + (2. * randomnumber());
            if (print)
                std::cout << l << "  ";
        }
        if (print)
            std::cout << "  P1 = ";
        for (double& j : triP1) {
            j = -1. + (2. * randomnumber());  // = fP0 + fN1*fA
            if (print)
                std::cout << j << "  ";
        }
        if (print)
            std::cout << "  P2 = ";
        for (double& k : triP2) {
            k = -1. + (2. * randomnumber());  // = fP0 + fN2*fB
            if (print)
                std::cout << k << "  ";
        }
        std::cout << std::endl;

        ////////////////////
        // TRIANGLE ARRAY //
        ////////////////////

        double triData[11];

        // compute further triangle data
        triData[0] = sqrt(POW2(triP1[0] - triP0[0]) + POW2(triP1[1] - triP0[1]) + POW2(triP1[2] - triP0[2]));
        triData[1] = sqrt(POW2(triP2[0] - triP0[0]) + POW2(triP2[1] - triP0[1]) + POW2(triP2[2] - triP0[2]));
        triData[2] = triP0[0];
        triData[3] = triP0[1];
        triData[4] = triP0[2];
        triData[5] = (triP1[0] - triP0[0]) / triData[0];
        triData[6] = (triP1[1] - triP0[1]) / triData[0];
        triData[7] = (triP1[2] - triP0[2]) / triData[0];
        triData[8] = (triP2[0] - triP0[0]) / triData[1];
        triData[9] = (triP2[1] - triP0[1]) / triData[1];
        triData[10] = (triP2[2] - triP0[2]) / triData[1];

        // get perpendicular normal vector n3 on triangle surface

        double triN3[3];
        triN3[0] = triData[6] * triData[10] - triData[7] * triData[9];
        triN3[1] = triData[7] * triData[8] - triData[5] * triData[10];
        triN3[2] = triData[5] * triData[9] - triData[6] * triData[8];
        const double triMagN3 = 1. / sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
        triN3[0] = triN3[0] * triMagN3;
        triN3[1] = triN3[1] * triMagN3;
        triN3[2] = triN3[2] * triMagN3;

        // triangle centroid

        // THIS IS NEVER USED
        /*const double triCenter[3] = {
                triData[2] + (triData[0]*triData[5] + triData[1]*triData[8])/3.,
                triData[3] + (triData[0]*triData[6] + triData[1]*triData[9])/3.,
                triData[4] + (triData[0]*triData[7] + triData[1]*triData[10])/3.};*/

        // side line vectors

        const double triAlongSideP0P1[3] = {triData[0] * triData[5],
                                            triData[0] * triData[6],
                                            triData[0] * triData[7]};  // = A * N1

        const double triAlongSideP1P2[3] = {triP2[0] - triP1[0], triP2[1] - triP1[1], triP2[2] - triP1[2]};

        const double triAlongSideP2P0[3] = {(-1) * triData[1] * triData[8],
                                            (-1) * triData[1] * triData[9],
                                            (-1) * triData[1] * triData[10]};  // = -B * N2

        // length values of side lines

        const double triAlongSideLengthP0P1 = triData[0];
        const double triAlongSideLengthP1P2 =
            sqrt(POW2(triP2[0] - triP1[0]) + POW2(triP2[1] - triP1[1]) + POW2(triP2[2] - triP1[2]));
        const double triAlongSideLengthP2P0 = triData[1];

        // side line unit vectors

        double triAlongSideP0P1Unit[3] = {triData[5], triData[6], triData[7]};  // = N1
        double triAlongSideP1P2Unit[3];

        const double magP1P2 =
            1. / sqrt(POW2(triP2[0] - triP1[0]) + POW2(triP2[1] - triP1[1]) + POW2(triP2[2] - triP1[2]));
        triAlongSideP1P2Unit[0] = magP1P2 * (triP2[0] - triP1[0]);
        triAlongSideP1P2Unit[1] = magP1P2 * (triP2[1] - triP1[1]);
        triAlongSideP1P2Unit[2] = magP1P2 * (triP2[2] - triP1[2]);

        double triAlongSideP2P0Unit[3] = {-triData[8], -triData[9], -triData[10]};  // = -N2

        /////////////////////
        // TRIANGLE OBJECT //
        /////////////////////

        auto* triangle = new KEMTriangle();
        triangle->SetA(triData[0]);
        triangle->SetB(triData[1]);
        triangle->SetP0(KFieldVector(triP0[0], triP0[1], triP0[2]));
        triangle->SetN1(KFieldVector(triData[5], triData[6], triData[7]));
        triangle->SetN2(KFieldVector(triData[8], triData[9], triData[10]));

        triangle->SetBoundaryValue(1.);
        triangle->SetSolution(1.);

        //////////////////////
        // DICE FIELD POINT //
        //////////////////////

        double dirN1 = triData[0] * randomnumber();
        double dirN2 = triData[1] * randomnumber();

        // get component in N3 direction
        double averageSideLength = (triAlongSideLengthP0P1 + triAlongSideLengthP1P2 + triAlongSideLengthP2P0) / 3.;
        double dirN3 = 2. * averageSideLength * randomnumber();

        KFieldVector fP(0., 0., 0.);
        fP = KFieldVector(triData[2], triData[3], triData[4]) +
             (dirN1 * KFieldVector(triData[5], triData[6], triData[7])) +
             (dirN2 * KFieldVector(triData[8], triData[9], triData[10]));

        if (print)
            std::cout << "fP = " << fP[0] << "  " << fP[1] << "  " << fP[2] << std::endl;

        KFieldVector fPN3(0., 0., 0.);
        fPN3 = fP + (dirN3 * (triangle->GetN3()));

        // compute distance ratio
        // double dr = (triangle->Centroid() - fP).Magnitude() / averageSideLength;
        // FOR WHAT IF IT IS NEVER USED?

        // distance between triangle vertex points and field point IN PLANE !
        // in positive rotation order
        // pointing to the triangle vertex point

        const double triDistP0[3] = {triP0[0] - fP[0], triP0[1] - fP[1], triP0[2] - fP[2]};
        const double triDistP1[3] = {triP1[0] - fP[0], triP1[1] - fP[1], triP1[2] - fP[2]};
        const double triDistP2[3] = {triP2[0] - fP[0], triP2[1] - fP[1], triP2[2] - fP[2]};

        ////////////////////
        // DISTANCE CHECK //
        ////////////////////

        // check distance of field point to side line

        double distToLine = 0.;
        double lineLambda = -1.; /* parameter for distance vector */

        // auxiliary values for distance check

        double tmpVector[3];
        double tmpScalar;

        // 0 - check distances to P0P1 side line

        tmpScalar = 1. / triAlongSideLengthP0P1;

        // compute cross product

        tmpVector[0] = (triAlongSideP0P1[1] * triDistP0[2]) - (triAlongSideP0P1[2] * triDistP0[1]);
        tmpVector[1] = (triAlongSideP0P1[2] * triDistP0[0]) - (triAlongSideP0P1[0] * triDistP0[2]);
        tmpVector[2] = (triAlongSideP0P1[0] * triDistP0[1]) - (triAlongSideP0P1[1] * triDistP0[0]);

        distToLine = sqrt(POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2])) * tmpScalar;
        // initialization
        double distToLineMin = distToLine;
        int lineIndex = -1;  // identify side line

        // factor -1 in order to use array triDistP0
        lineLambda = ((-triDistP0[0] * triAlongSideP0P1[0]) + (-triDistP0[1] * triAlongSideP0P1[1]) +
                      (-triDistP0[2] * triAlongSideP0P1[2])) *
                     POW2(tmpScalar);
        if (print)
            std::cout << distToLine << "   ";
        if (print)
            std::cout << lineLambda << "   ";
        if (lineLambda >= -fToleranceLambda && lineLambda <= (1. + fToleranceLambda)) {
            distToLineMin = distToLine;
            lineIndex = 0;
        } /* lambda */

        // 1 - check distances to P1P2 side line

        tmpScalar = 1. / triAlongSideLengthP1P2;

        // compute cross product

        tmpVector[0] = (triAlongSideP1P2[1] * triDistP1[2]) - (triAlongSideP1P2[2] * triDistP1[1]);
        tmpVector[1] = (triAlongSideP1P2[2] * triDistP1[0]) - (triAlongSideP1P2[0] * triDistP1[2]);
        tmpVector[2] = (triAlongSideP1P2[0] * triDistP1[1]) - (triAlongSideP1P2[1] * triDistP1[0]);

        distToLine = sqrt(POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2])) * tmpScalar;
        // factor -1 for direction triDistP1 vector

        lineLambda = ((-triDistP1[0] * triAlongSideP1P2[0]) + (-triDistP1[1] * triAlongSideP1P2[1]) +
                      (-triDistP1[2] * triAlongSideP1P2[2])) *
                     POW2(tmpScalar);
        if (print)
            std::cout << distToLine << "   ";
        if (print)
            std::cout << lineLambda << "   ";
        if (lineLambda >= -fToleranceLambda && lineLambda <= (1. + fToleranceLambda)) {
            if (distToLine < distToLineMin) {
                distToLineMin = distToLine;
                lineIndex = 1;
            }
        } /* lambda */

        // 2 - check distances to P2P0 side line

        tmpScalar = 1. / triAlongSideLengthP2P0;

        // compute cross product

        tmpVector[0] = (triAlongSideP2P0[1] * triDistP2[2]) - (triAlongSideP2P0[2] * triDistP2[1]);
        tmpVector[1] = (triAlongSideP2P0[2] * triDistP2[0]) - (triAlongSideP2P0[0] * triDistP2[2]);
        tmpVector[2] = (triAlongSideP2P0[0] * triDistP2[1]) - (triAlongSideP2P0[1] * triDistP2[0]);

        distToLine = sqrt(POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2])) * tmpScalar;
        // factor -1 for triDistP2

        lineLambda = ((-triDistP2[0] * triAlongSideP2P0[0]) + (-triDistP2[1] * triAlongSideP2P0[1]) +
                      (-triDistP2[2] * triAlongSideP2P0[2])) *
                     POW2(tmpScalar);
        if (print)
            std::cout << distToLine << "   ";
        if (print)
            std::cout << lineLambda << "   ";
        if (lineLambda >= -fToleranceLambda && lineLambda <= (1. + fToleranceLambda)) {
            if (distToLine < distToLineMin) {
                distToLineMin = distToLine;
                lineIndex = 2;
            }
        } /* lambda */
        if (print)
            std::cout << "final min distance " << distToLineMin << " with index " << lineIndex << std::endl;

        // subdivide triangle and define new KEMTriangles, dependent from field point

        // outward pointing vector m, perpendicular to side lines (direction of min distance point)

        const double m0[3] = {(triAlongSideP0P1Unit[1] * triN3[2]) - (triAlongSideP0P1Unit[2] * triN3[1]),
                              (triAlongSideP0P1Unit[2] * triN3[0]) - (triAlongSideP0P1Unit[0] * triN3[2]),
                              (triAlongSideP0P1Unit[0] * triN3[1]) - (triAlongSideP0P1Unit[1] * triN3[0])};

        const double m1[3] = {(triAlongSideP1P2Unit[1] * triN3[2]) - (triAlongSideP1P2Unit[2] * triN3[1]),
                              (triAlongSideP1P2Unit[2] * triN3[0]) - (triAlongSideP1P2Unit[0] * triN3[2]),
                              (triAlongSideP1P2Unit[0] * triN3[1]) - (triAlongSideP1P2Unit[1] * triN3[0])};

        const double m2[3] = {(triAlongSideP2P0Unit[1] * triN3[2]) - (triAlongSideP2P0Unit[2] * triN3[1]),
                              (triAlongSideP2P0Unit[2] * triN3[0]) - (triAlongSideP2P0Unit[0] * triN3[2]),
                              (triAlongSideP2P0Unit[0] * triN3[1]) - (triAlongSideP2P0Unit[1] * triN3[0])};

        // m = unit vector as cross product of two perpendicular unit vectors

        KFieldVector goOut;

        // decide which m has to be taken
        if (lineIndex == 0)
            goOut = KFieldVector(m0[0], m0[1], m0[2]);  // Unit vectors ???????
        if (lineIndex == 1)
            goOut = KFieldVector(m1[0], m1[1], m1[2]);
        if (lineIndex == 2)
            goOut = KFieldVector(m2[0], m2[1], m2[2]);

        KFieldVector goAlong;

        if (lineIndex == 0)
            goAlong = KFieldVector(triAlongSideP0P1Unit[0], triAlongSideP0P1Unit[1], triAlongSideP0P1Unit[2]);
        if (lineIndex == 1)
            goAlong = KFieldVector(triAlongSideP1P2Unit[0], triAlongSideP1P2Unit[1], triAlongSideP1P2Unit[2]);
        if (lineIndex == 2)
            goAlong = KFieldVector(triAlongSideP2P0Unit[0], triAlongSideP2P0Unit[1], triAlongSideP2P0Unit[2]);

        // save rectangle points

        std::vector<KFieldVector> rect;

        rect.push_back(fP + (distToLineMin * goOut));
        rect.push_back(fP + (distToLineMin * goAlong));
        rect.push_back(fP - (distToLineMin * goOut));
        rect.push_back(fP - (distToLineMin * goAlong));


        // define triangle corner points relatively to side with min distance

        KFieldVector p0Rel;
        KFieldVector p1Rel;
        KFieldVector p2Rel;

        if (lineIndex == 0) {
            p0Rel = triP0;
            p1Rel = triP1;
            p2Rel = triP2;
        }
        if (lineIndex == 1) {
            p0Rel = triP1;
            p1Rel = triP2;
            p2Rel = triP0;
        }
        if (lineIndex == 2) {
            p0Rel = triP2;
            p1Rel = triP0;
            p2Rel = triP1;
        }

        // define six new triangles, common n3 direction

        std::vector<KEMTriangle> triCont;
        AddTriangle(triCont, rect[0], p1Rel, rect[1]);  // 1
        AddTriangle(triCont, rect[1], p1Rel, p2Rel);    // 2
        AddTriangle(triCont, rect[2], rect[1], p2Rel);  // 3
        AddTriangle(triCont, rect[3], rect[2], p2Rel);  // 4
        AddTriangle(triCont, rect[3], p2Rel, p0Rel);    // 5
        AddTriangle(triCont, rect[0], rect[3], p0Rel);  // 6

        // compute field/potential via Gauss Legendre and sum up values !!

        double triPotential = 0.;
        KFieldVector triField;

        for (auto& i : triCont) {
            triPotential += intTriQuad.Potential(i.GetShape(), fP);
            triField += intTriQuad.ElectricField(i.GetShape(), fP);
        }

        // add field and potential for rectangle

        // check z and compute omega, if z=0 omega = 2.Pi
        // WTF this is never used
        /*double omega = 2.*M_PI;
        if( dirN3>1.E-14 ) {
            omega = 0.; // compute
        }*/
    }


    // compute Ez

    // compute RWG integral, potential and field

    // compute RWG relative error

    // compute analytical integral, potential and field

    // compute analytical relative error

    // write two separate graphs for relative errors

    // graph for error differences of analytical and rwg


    return 0;
}

void subrn(double* u, int len)
{
    // This subroutine computes random numbers u[1],...,u[len]
    // in the (0,1) interval. It uses the 0<IJKLRANDOM<900000000
    // integer as initialization seed.
    //  In the calling program the dimension
    // of the u[] vector should be larger than len (the u[0] value is
    // not used).
    // For each IJKLRANDOM
    // numbers the program computes completely independent random number
    // sequences (see: F. James, Comp. Phys. Comm. 60 (1990) 329, sec. 3.3).

    static int iff = 0;
    static long ijkl, ij, kl, i, j, k, l, ii, jj, m, i97, j97, ivec;
    static float s, t, uu[98], c, cd, cm, uni;
    if (iff == 0) {
        if (IJKLRANDOM == 0) {
            std::cout << "Message from subroutine subrn:\n";
            std::cout << "the global integer IJKLRANDOM should be larger than 0 !!!\n";
            std::cout << "Computation is  stopped !!! \n";
            exit(0);
        }
        ijkl = IJKLRANDOM;
        if (ijkl < 1 || ijkl >= 900000000)
            ijkl = 1;
        ij = ijkl / 30082;
        kl = ijkl - 30082 * ij;
        i = ((ij / 177) % 177) + 2;
        j = (ij % 177) + 2;
        k = ((kl / 169) % 178) + 1;
        l = kl % 169;
        for (ii = 1; ii <= 97; ii++) {
            s = 0;
            t = 0.5;
            for (jj = 1; jj <= 24; jj++) {
                m = (((i * j) % 179) * k) % 179;
                i = j;
                j = k;
                k = m;
                l = (53 * l + 1) % 169;
                if ((l * m) % 64 >= 32)
                    s = s + t;
                t = 0.5 * t;
            }
            uu[ii] = s;
        }
        c = 362436. / 16777216.;
        cd = 7654321. / 16777216.;
        cm = 16777213. / 16777216.;
        i97 = 97;
        j97 = 33;
        iff = 1;
    }
    for (ivec = 1; ivec <= len; ivec++) {
        uni = uu[i97] - uu[j97];
        if (uni < 0.)
            uni = uni + 1.;
        uu[i97] = uni;
        i97 = i97 - 1;
        if (i97 == 0)
            i97 = 97;
        j97 = j97 - 1;
        if (j97 == 0)
            j97 = 97;
        c = c - cd;
        if (c < 0.)
            c = c + cm;
        uni = uni - c;
        if (uni < 0.)
            uni = uni + 1.;
        if (uni == 0.) {
            uni = uu[j97] * 0.59604644775391e-07;
            if (uni == 0.)
                uni = 0.35527136788005e-14;
        }
        u[ivec] = uni;
    }
}

////////////////////////////////////////////////////////////////

double randomnumber()
{
    // This function computes 1 random number in the (0,1) interval,
    // using the subrn subroutine.

    double u[2];
    subrn(u, 1);
    return u[1];
}
