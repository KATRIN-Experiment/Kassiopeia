// TestDielectrics.cc (adapted from KEMField 1.0)
// Daniel Hilk, 22.10.2017

// Geometric model:
// * Box filled with liquid and gaseous Xenon, eps_r(LXe) = 2, e_r(GXe) = 1.
// * Two electrodes (either plates or parallel wires).
//
// Output:
// * ROOT TFile containing four TGraphs (potential, electric field components E_z and E_r, electric displacement D_z)
// * All values will be computed along z-axis at x=y=0.

// c++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// kemfield
#include "KBinaryDataStreamer.hh"
#include "KChargeDensitySolver.hh"
#include "KDataDisplay.hh"
#include "KEMConstants.hh"
#include "KEMFileInterface.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KSADataStreamer.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"


// linear algebra
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KGaussianElimination.hh"
#include "KIterationTracker.hh"
#include "KIterativeStateWriter.hh"
#include "KMultiElementRobinHood.hh"
#include "KRobinHood.hh"
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KRobinHood_OpenCL.hh"
using KEMField::KRobinHood_OpenCL;
#endif


// field solver
#include "KElectrostaticIntegratingFieldSolver.hh"
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif

// root
#include "Rtypes.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TMath.h"
#include "TSystem.h"
#include "TVector3.h"


using namespace KEMField;
using namespace std;


// typedefs for Dirichlet and Neumann elements
typedef KSurface<KElectrostaticBasis, KNeumannBoundary, KRectangle> KEMBoundary;
using KEMBoundaryTriangle = KSurface<KElectrostaticBasis, KNeumannBoundary, KTriangle>;
using KEMRectangle = KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>;
using KEMWire = KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment>;


// Multi-Element Robin Hood
#define MULTIRH

// Use triangles instead of rectangles for Neumann boundary
#define USENEUMANNTRI

bool StrToBool(const std::string& s)
{
    bool b;
    std::stringstream ss(s);  //turn the string into a stream
    ss >> b;                  //convert
    return b;
}

/**
 * This function takes an interval of length <interval> and discretizes it
 * into <nSegments> # of segments, with the size distribution determined by
 * <power>.  The resulting <nSegments> intervals are recorded into the array
 * <segments>.
 *
 * <mid> is the lenth of <interval> that is mirrored.  If <nSegments> is odd,
 * then the largest segment is in the middle of the original interval, and the
 * remaining lenth of <interval> is mirrored.
 */
void DiscretizeInterval(double interval, int nSegments, double power, std::vector<double>& segments)
{
    if (nSegments == 1)
        segments[0] = interval;
    else {
        double inc1, inc2;
        double mid = interval * .5;
        if (nSegments % 2 == 1) {
            segments[nSegments / 2] = interval / nSegments;
            mid -= interval / (2 * nSegments);
        }

        for (Int_t i = 0; i < nSegments / 2; i++) {
            inc1 = ((double) i) / (nSegments / 2);
            inc2 = ((double) (i + 1)) / (nSegments / 2);

            inc1 = pow(inc1, power);
            inc2 = pow(inc2, power);

            segments[i] = segments[nSegments - (i + 1)] = mid * (inc2 - inc1);
        }
    }
}


/**
 * Adds the rectangle to the surface container and Discretizes
 * the rectangle into smaller rectangles, where the
 * approximation of constant charge density is more reasonable.
 */
void AddRect(KSurfaceContainer& fContainer, int& fGroup, int& fChDen, double fA, double fB, KFieldVector fP0,
             KFieldVector fN1, KFieldVector fN2, double fU, /* potential */
             double fNRot, int fNumDiscA, int fNumDiscB)
{
    fNRot++;
    fChDen += (fNumDiscA * fNumDiscB);

    // do not discretize if discretization parameters are set to 0
    if (fNumDiscA == 0 || fNumDiscB == 0) {
        auto* rectangle = new KEMRectangle();

        rectangle->SetA(fA);
        rectangle->SetB(fB);
        rectangle->SetP0(fP0);
        rectangle->SetN1(fN1);
        rectangle->SetN2(fN2);

        rectangle->SetBoundaryValue(fU);

        fContainer.push_back(rectangle);

        fGroup++;
        return;
    }

    const double scale = 1.;
    const double power = 2.;

    // rescale the discretization parameters by GetDiscScale()
    fNumDiscA *= scale;
    fNumDiscB *= scale;

    // vectors a,b contain the lengths of the rectangles
    std::vector<double> a(fNumDiscA);
    std::vector<double> b(fNumDiscB);

    DiscretizeInterval(fA, fNumDiscA, power, a);
    DiscretizeInterval(fB, fNumDiscB, power, b);

    // dA and db are the offsets for each fP0[3] cornerpoint
    double dA = 0;
    double dB = 0;

    for (int i = 0; i < fNumDiscA; i++) {
        dB = 0;

        for (int j = 0; j < fNumDiscB; j++) {
            auto* rectangle = new KEMRectangle();
            rectangle->SetN1(fN1);
            rectangle->SetN2(fN2);
            rectangle->SetBoundaryValue(fU);

            // set length A
            rectangle->SetA(a[i]);
            // set P0
            KPosition p0New((fP0[0] + dA * fN1[0] + dB * fN2[0]),
                            (fP0[1] + dA * fN1[1] + dB * fN2[1]),
                            (fP0[2] + dA * fN1[2] + dB * fN2[2]));
            rectangle->SetP0(p0New);

            //std::cout << "x: " << p0New[0] << " y: " << p0New[1] << " z: " << p0New[2] << std::endl;

            // set length B
            rectangle->SetB(b[j]);
            dB += b[j];

            // add r to the surface container
            fContainer.push_back(rectangle);
        } /*B direction*/
        dA += a[i];
    } /*A direction*/

    //std::cout << fContainer.size() << std::endl;

    fGroup++;
}


/**
 * Adds the wire to the surface container and discretizes
 * the wire into smaller wires, where the approximation of
 * constant charge density is more reasonable.
 */
void AddWire(KSurfaceContainer& fContainer, int& fGroup, int& fChDen, KPosition fPA, KPosition fPB,
             double fD /*diameter*/, double fU /*potential*/, int fNRot, int fNumDisc)
{
    (void) fNRot;
    fChDen += fNumDisc;

    // do not discretize if discretization parameter is set to 0
    if (fNumDisc == 0) {
        auto* wire = new KEMWire();

        wire->SetP0(fPA);
        wire->SetP1(fPB);
        wire->SetDiameter(fD);
        wire->SetBoundaryValue(fU);
        fContainer.push_back(wire);

        fGroup++;
        return;
    }

    double A[3] = {0, 0, 0};  // new wire parameters
    double B[3] = {0, 0, 0};

    const double scale = 1.;
    const double power = 2.;

    int nIncrements = scale * fNumDisc;

    std::vector<std::vector<double>> inc(3, std::vector<double>(nIncrements, 0));

    do {
        DiscretizeInterval((fPB[0] - fPA[0]), nIncrements, power, inc[0]);
        DiscretizeInterval((fPB[1] - fPA[1]), nIncrements, power, inc[1]);
        DiscretizeInterval((fPB[2] - fPA[2]), nIncrements, power, inc[2]);
        nIncrements--;
    } while (sqrt(inc[0][0] * inc[0][0] + inc[1][0] * inc[1][0] + inc[2][0] * inc[2][0]) < fD);
    nIncrements++;


    for (int i = 0; i < 3; i++)
        B[i] = fPA[i];

    for (int i = 0; i < nIncrements / 2; i++) {
        auto* wire = new KEMWire();
        wire->SetDiameter(fD);
        wire->SetBoundaryValue(fU);

        // loop over components
        for (int j = 0; j < 3; j++) {
            A[j] = B[j];
            B[j] += inc[j][i];

            fPA[j] = A[j];
            fPB[j] = B[j];
        }
        wire->SetP0(fPA);
        wire->SetP1(fPB);
        fContainer.push_back(wire);
    }

    for (int i = 0; i < 3; i++)
        A[i] = fPB[i];

    // Change: Added "-1" to calculate with the right number of wires.
    for (int i = nIncrements - 1; i >= nIncrements / 2; i--) {
        auto* wire = new KEMWire();
        wire->SetDiameter(fD);
        wire->SetBoundaryValue(fU);

        // loop over components
        for (int j = 0; j < 3; j++) {
            B[j] = A[j];
            A[j] -= inc[j][i];

            fPA[j] = A[j];
            fPB[j] = B[j];
        }
        wire->SetP0(fPA);
        wire->SetP1(fPB);
        fContainer.push_back(wire);
    }

    fGroup++;
}

#ifdef USENEUMANNTRI
void AddBoundary(KSurfaceContainer& fContainer, int& fGroup, int& fChDen, double fA, double fB, const KFieldVector& fP0,
                 const KFieldVector& fN1, const KFieldVector& fN2, double fEpsRAbove, double fEpsRBelow, double fNRot,
                 int fNumDiscA, int fNumDiscB)
{
    (void) fNRot;
    fChDen += (fNumDiscA * fNumDiscB);

    // do not discretize if discretization parameters are set to 0
    if (fNumDiscA == 0 || fNumDiscB == 0) {
        std::cout << "No discretization parameter set! No boundary element added." << std::endl;

        return;
    }

    const double scale = 1.;
    const double power = 2.;

    //  rescale the discretization parameters by GetDiscScale()
    fNumDiscA *= scale;
    fNumDiscB *= scale;

    // vectors a,b contain the lengths of the rectangles
    std::vector<Double_t> a(fNumDiscA);
    std::vector<Double_t> b(fNumDiscB);

    DiscretizeInterval(fA, fNumDiscA, power, a);
    DiscretizeInterval(fB, fNumDiscB, power, b);

    // dA and db are the offsets for each fP0[3] cornerpoint
    double dA = 0;
    double dB = 0;

    for (int i = 0; i < fNumDiscA; i++) {
        dB = 0;

        for (int j = 0; j < fNumDiscB; j++) {
            auto* tri1 = new KEMBoundaryTriangle();
            auto* tri2 = new KEMBoundaryTriangle();

            tri1->SetP0(fP0 + dA * fN1 + dB * fN2);
            tri1->SetN1(fN1);
            tri1->SetN2(fN2);
            tri2->SetP0(fP0 + (dA + a[i]) * fN1 + (dB + b[j]) * fN2);
            tri2->SetN1(-1. * fN1);
            tri2->SetN2(-1. * fN2);

            // switch N1 and N2 if necessary for positive n3.Z
            if (tri1->GetN3().Z() < 0) {
                std::cout << "The normal vectors N1 and N2 have been exchanged for positive N3.\n";
                tri1->SetN1(fN2);
                tri1->SetN2(fN1);
                tri1->SetN3();
            }
            if (tri2->GetN3().Z() < 0) {
                std::cout << "The normal vectors N1 and N2 have been exchanged for positive N3.\n";
                tri2->SetN1(fN2);
                tri2->SetN2(fN1);
                tri2->SetN3();
            }

            tri1->SetNormalBoundaryFlux(fEpsRBelow / fEpsRAbove);
            tri2->SetNormalBoundaryFlux(fEpsRBelow / fEpsRAbove);

            // set length A
            tri1->SetA(a[i]);
            tri2->SetA(a[i]);

            // set length B
            tri1->SetB(b[j]);
            tri2->SetB(b[j]);

            // add tri1 and tri2 to the surface container
            fContainer.push_back(tri1);
            fContainer.push_back(tri2);

            dB += b[j];
        } /*B direction*/
        dA += a[i];
    } /*A direction*/

    fGroup++;
}
#else
void AddBoundary(KSurfaceContainer& fContainer, int& fGroup, int& fChDen, double fA, double fB, KFieldVector fP0,
                 KFieldVector fN1, KFieldVector fN2, double fEpsRAbove, double fEpsRBelow, double fNRot, int fNumDiscA,
                 int fNumDiscB)
{
    (void) fNRot;
    fChDen += (fNumDiscA * fNumDiscB);

    // do not discretize if discretization parameters are set to 0
    if (fNumDiscA == 0 || fNumDiscB == 0) {
        KEMBoundary* rectangle = new KEMBoundary();

        rectangle->SetA(fA);
        rectangle->SetB(fB);
        rectangle->SetP0(fP0);
        rectangle->SetN1(fN1);
        rectangle->SetN2(fN2);
        rectangle->SetN3();

        // switch N1 and N2 if necessary for positive n3.Z
        if (rectangle->GetN3().Z() < 0) {
            rectangle->SetN1(fN2);
            rectangle->SetN2(fN1);
            rectangle->SetN3();
        }

        rectangle->SetNormalBoundaryFlux(fEpsRBelow / fEpsRAbove);

        fContainer.push_back(rectangle);

        fGroup++;
        return;
    }

    const double scale = 1.;
    const double power = 2.;

    //  rescale the discretization parameters by GetDiscScale()
    fNumDiscA *= scale;
    fNumDiscB *= scale;

    // vectors a,b contain the lengths of the rectangles
    std::vector<Double_t> a(fNumDiscA);
    std::vector<Double_t> b(fNumDiscB);

    DiscretizeInterval(fA, fNumDiscA, power, a);
    DiscretizeInterval(fB, fNumDiscB, power, b);

    // dA and db are the offsets for each fP0[3] cornerpoint
    double dA = 0;
    double dB = 0;

    for (int i = 0; i < fNumDiscA; i++) {

        dB = 0;

        for (int j = 0; j < fNumDiscB; j++) {
            KEMBoundary* rectangle = new KEMBoundary();
            rectangle->SetN1(fN1);
            rectangle->SetN2(fN2);
            rectangle->SetN3();
            // switch N1 and N2 if necessary for positive n3.Z
            if (rectangle->GetN3().Z() < 0) {
                std::cout << "The normal vectors N1 and N2 have been exchanged for positive N3.\n";
                rectangle->SetN1(fN2);
                rectangle->SetN2(fN1);
                rectangle->SetN3();
            }

            rectangle->SetNormalBoundaryFlux(fEpsRBelow / fEpsRAbove);

            // set length A
            rectangle->SetA(a[i]);
            // set P0
            KPosition p0New((fP0[0] + dA * fN1[0] + dB * fN2[0]),
                            (fP0[1] + dA * fN1[1] + dB * fN2[1]),
                            (fP0[2] + dA * fN1[2] + dB * fN2[2]));
            rectangle->SetP0(p0New);

            // set length B
            rectangle->SetB(b[j]);
            dB += b[j];

            // add r to the surface container
            fContainer.push_back(rectangle);
        } /*B direction*/
        dA += a[i];
    } /*A direction*/

    fGroup++;
    return;
}
#endif

KFieldVector CalcTime(double fDuration)
{
    // Calculating a time, given in seconds, in hours, minutes and seconds.
    // Result will be written into a vector.
    KFieldVector fTime;

    int fSecondsTotal(fDuration);

    int fSecondsRest = fSecondsTotal % 60;
    int fMinutesTotal = fSecondsTotal / 60;
    fTime.SetX(fSecondsRest);

    int fMinutesRest = fMinutesTotal % 60;
    fTime.SetY(fMinutesRest);

    int fHoursTotal = fMinutesTotal / 60;
    fTime.SetZ(fHoursTotal);

    return fTime;
}


int main(int argc, char* argv[])
{
    std::cout << "________________________________________________________________________________" << std::endl
              << std::endl;
    std::cout << "           ELECTRIC FIELD CALCULATION PROGRAM FOR TEST OF DIELECTRICS           " << std::endl;
    std::cout << "________________________________________________________________________________" << std::endl
              << std::endl;

    // ----------------------------------------------------------------------------

    // Technical definitions:
    std::string fModelName;
    std::string fMainDir;
    std::string ftextinput;
    int fGroupIndex(0);
    int fChDensities(0);
    int k(0);
    bool fFixedPoints(false);
    (void) fFixedPoints;
    KFieldVector gLocation(0., 0., -0.25);
    int gSteps(5000);
    double gStepsize(0.0001);
    vector<double> v;
    vector<double> c;

    if (argc >= 3) {
        fFixedPoints = true;
        ftextinput = argv[2];
    }
    else {
        fFixedPoints = false;
    }

    fMainDir = (std::string) gSystem->GetFromPipe("pwd") + "/";
    fModelName = "TestName" + (std::string) argv[1];
    int disc(10);
    std::cout << fMainDir << std::endl;

    std::string fElectrodeDir(fMainDir + "KEM_" + fModelName + ".root");
    std::string fFieldOutput(fMainDir + "ESTATICS_" + fModelName + ".root");


    // ----------------------------------------------------------------------------

    // Physical input parameters:
    double fAnodeU(1500.);    // Voltage of anode (top)
    double fCathodeU(-500.);  // Voltage of cathode (bottom)
    double fBoxU(0.e0);
    double fGXeEpsR(1.);
    double fLXeEpsR(2.);
    // double fGXeEpsR(1.00126e0); // Dielectric constant for upper part (gaseous Xenon)
    // double fLXeEpsR(1.96e0); // Dielectric constant for lower part (liquid Xenon)

    // ----------------------------------------------------------------------------

    // Geometrical setup:
    bool fAnode(true);
    bool fCathode(true);

    bool fBoundary(true);

    bool fUseWires(false);              // if true, then use parallel wires as electrodes, otherwise use plates
    bool fConnectWireEndPoints(false);  // Connect open end points of parallel anode and cathode wires

    bool fBoxXY(false);
    bool fBoxYZ(false);
    bool fBoxZX(false);

    // ----------------------------------------------------------------------------

    // Geometric input parameters:
    int fWiresPerLayer(26);  // please only even numbers!
    double fWirePitchY(1.e-2);
    double fWireDiameter(1.e-3);
    // Additional space between main and short connection wire parts:
    double fWireConnectDistX(5.e-3);
    if (!fConnectWireEndPoints)
        fWireConnectDistX = 0.e0;

    // Calculate fWireLengthX from fWirePitchY, fWireDiameter
    // and fWiresPerLayer, to ensure a quadratic area.
    double fWireLengthX((fWiresPerLayer * fWireDiameter) + ((fWiresPerLayer - 1) * fWirePitchY));

    double fBoundaryHeightZ(0.e0);
    double fLayerDistanceZ(20.e-3);
    fLayerDistanceZ = fLayerDistanceZ / 2.e0;

    // Use only half of fWireLengthX due to technical reasons:
    fWireLengthX = fWireLengthX / 2.e0;

    // Use only half of fWirePerLayer due to technical reasons:
    fWiresPerLayer = fWiresPerLayer / 2;

    double fBoxLengthX(400.e-3);
    fBoxLengthX = fBoxLengthX / 2.e0;
    double fBoxWidthY(400.e-3);
    fBoxWidthY = fBoxWidthY / 2.e0;
    double fBoxHeightZ(400.e-3);
    fBoxHeightZ = fBoxHeightZ / 2.e0;

    // For loop over y-axis to define wire electrodes:
    double fRunAbsY(0.e0);
    double fStepY((fWirePitchY + fWireDiameter) / 2.e0);

    // Length and width of different rectangles in XY-layer:
    double fXY269WidthY((((2 * fWiresPerLayer) - 1) * 2.e0 * fStepY) + fWireDiameter);
    double fXY134578WidthY(fBoxWidthY - (fXY269WidthY / 2.e0));
    double fXY489LengthX((2.e0 * (fWireLengthX + fWireConnectDistX)) + fWireDiameter);
    double fXY123567LengthX(fBoxLengthX - (fXY489LengthX / 2.e0));

    // Length and width of different rectangles in YZ-layer:
    double fYZ123HeightZ(fLayerDistanceZ);
    double fYZ456HeightZ(fBoxHeightZ - fLayerDistanceZ);
    double fYZ1346WidthY(fXY134578WidthY);
    double fYZ25WidthY(fXY269WidthY);

    // Length and width of different rectangles in ZX-layer:
    double fZX1346LengthX(fXY123567LengthX);
    double fZX25LengthX(fXY489LengthX);
    double fZX123HeightZ(fYZ123HeightZ);
    double fZX456HeightZ(fYZ456HeightZ);

    // ----------------------------------------------------------------------------

    // Start- and endpoints:
    KFieldVector fWireAnodePA(fWireLengthX, fRunAbsY, fLayerDistanceZ);
    KFieldVector fWireAnodePB(-fWireLengthX, fRunAbsY, fLayerDistanceZ);
    KFieldVector fWireCathodePA(fWireLengthX, fRunAbsY, -fLayerDistanceZ);
    KFieldVector fWireCathodePB(-fWireLengthX, fRunAbsY, -fLayerDistanceZ);

    // End points of connecting wires:
    KFieldVector fWireConnectPA;
    KFieldVector fWireConnectPB;

    // Normal vectors:
    KFieldVector fNx(1., 0., 0.);
    KFieldVector fNy(0., 1., 0.);
    KFieldVector fNz(0., 0., 1.);

    // Common P0 vector for rectangles:
    KFieldVector fP0(0., 0., 0.);

    // ----------------------------------------------------------------------------

    // Discretization of wires:
    int fWireDisc(70);
    int fWireConnectDisc(5);

    // Discretization of one XY-layer:
    int fRectDiscXY1357(disc);
    int fRectDiscXY26(disc);
    int fRectDiscXY48(disc);
    int fRectDiscXY9(disc);

    // Discretization of the virtual boundary XY-layer:
    int fBoundaryDiscXY1357(disc);
    int fBoundaryDiscXY26(disc);
    int fBoundaryDiscXY48(disc);
    int fBoundaryDiscXY9(disc);

    // Discretization of YZ-layer:
    int fRectDiscYZ13(disc);
    int fRectDiscYZ46(disc);
    int fRectDiscYZ2(disc);
    int fRectDiscYZ5(disc);

    // Discretization of ZX-layer:
    int fRectDiscZX13(disc);
    int fRectDiscZX46(disc);
    int fRectDiscZX2(disc);
    int fRectDiscZX5(disc);

    // Discretization of electrode plates (XY-layer):
    double fRectElectrodeDisc(disc);

    // ----------------------------------------------------------------------------

    KSurfaceContainer surfaceContainer;

    // ----------------------------------------------------------------------------


    if (fUseWires) {

        // The wires are parallel to the x-axis, the y-coordinate will be iterated.

        for (int i = 1; i <= fWiresPerLayer; i++) {

            fRunAbsY += fStepY;

            if (fAnode) {

                // Positive y-direction, y > 0
                fWireAnodePA.SetY(fRunAbsY);
                fWireAnodePB.SetY(fRunAbsY);
                AddWire(surfaceContainer,
                        fGroupIndex,
                        fChDensities,
                        fWireAnodePA,
                        fWireAnodePB,
                        fWireDiameter,
                        fAnodeU,
                        1,
                        fWireDisc);

                // Connect wire endpoints (in total 4 wire elements)
                if (fConnectWireEndPoints) {

                    // Front, x > 0
                    fWireConnectPA.SetComponents(fWireLengthX + fWireConnectDistX, fRunAbsY, fLayerDistanceZ);
                    fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX, fRunAbsY - fStepY, fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fAnodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                     fRunAbsY + fStepY,
                                                     fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fAnodeU,
                                1,
                                fWireConnectDisc);
                    }

                    // Back, x < 0
                    fWireConnectPA.SetComponents(-fWireLengthX - fWireConnectDistX, fRunAbsY, fLayerDistanceZ);
                    fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX, fRunAbsY - fStepY, fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fAnodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                     fRunAbsY + fStepY,
                                                     fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fAnodeU,
                                1,
                                fWireConnectDisc);
                    }
                }

                // Negative y-direction, y < 0
                fWireAnodePA.SetY(-fRunAbsY);
                fWireAnodePB.SetY(-fRunAbsY);
                AddWire(surfaceContainer,
                        fGroupIndex,
                        fChDensities,
                        fWireAnodePA,
                        fWireAnodePB,
                        fWireDiameter,
                        fAnodeU,
                        1,
                        fWireDisc);

                // Connect wire endpoints (in total 4 wire elements)
                if (fConnectWireEndPoints) {

                    // Front, x > 0
                    fWireConnectPA.SetComponents(fWireLengthX + fWireConnectDistX, -fRunAbsY, fLayerDistanceZ);
                    fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                 -(fRunAbsY - fStepY),
                                                 fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fAnodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                     -(fRunAbsY + fStepY),
                                                     fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fAnodeU,
                                1,
                                fWireConnectDisc);
                    }

                    // Back, x < 0
                    fWireConnectPA.SetComponents(-fWireLengthX - fWireConnectDistX, -fRunAbsY, fLayerDistanceZ);
                    fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                 -(fRunAbsY - fStepY),
                                                 fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fAnodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                     -(fRunAbsY + fStepY),
                                                     fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fAnodeU,
                                1,
                                fWireConnectDisc);
                    }
                }
            }

            if (fCathode) {

                // Positive y-direction, y > 0
                fWireCathodePA.SetY(fRunAbsY);
                fWireCathodePB.SetY(fRunAbsY);
                AddWire(surfaceContainer,
                        fGroupIndex,
                        fChDensities,
                        fWireCathodePA,
                        fWireCathodePB,
                        fWireDiameter,
                        fCathodeU,
                        1,
                        fWireDisc);

                // Connect wire endpoints (in total 4 wire elements)
                if (fConnectWireEndPoints) {

                    // Front, x > 0
                    fWireConnectPA.SetComponents(fWireLengthX + fWireConnectDistX, fRunAbsY, -fLayerDistanceZ);
                    fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX, fRunAbsY - fStepY, -fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fCathodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                     fRunAbsY + fStepY,
                                                     -fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fCathodeU,
                                1,
                                fWireConnectDisc);
                    }

                    // Back, x < 0
                    fWireConnectPA.SetComponents(-fWireLengthX - fWireConnectDistX, fRunAbsY, -fLayerDistanceZ);
                    fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                 fRunAbsY - fStepY,
                                                 -fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fCathodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                     fRunAbsY + fStepY,
                                                     -fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fCathodeU,
                                1,
                                fWireConnectDisc);
                    }
                }

                // Negative y-direction, y < 0
                fWireCathodePA.SetY(-fRunAbsY);
                fWireCathodePB.SetY(-fRunAbsY);
                AddWire(surfaceContainer,
                        fGroupIndex,
                        fChDensities,
                        fWireCathodePA,
                        fWireCathodePB,
                        fWireDiameter,
                        fCathodeU,
                        1,
                        fWireDisc);

                // Connect wire endpoints (in total 4 wire elements)
                if (fConnectWireEndPoints) {

                    // Front, x > 0
                    fWireConnectPA.SetComponents(fWireLengthX + fWireConnectDistX, -fRunAbsY, -fLayerDistanceZ);
                    fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                 -(fRunAbsY - fStepY),
                                                 -fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fCathodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                     -(fRunAbsY + fStepY),
                                                     -fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fCathodeU,
                                1,
                                fWireConnectDisc);
                    }

                    // Back, x < 0
                    fWireConnectPA.SetComponents(-fWireLengthX - fWireConnectDistX, -fRunAbsY, -fLayerDistanceZ);
                    fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                 -(fRunAbsY - fStepY),
                                                 -fLayerDistanceZ);
                    AddWire(surfaceContainer,
                            fGroupIndex,
                            fChDensities,
                            fWireConnectPA,
                            fWireConnectPB,
                            fWireDiameter,
                            fCathodeU,
                            1,
                            fWireConnectDisc);
                    if (i < fWiresPerLayer) {
                        // to ensure a quadratic area, the following step would be too much
                        fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                     -(fRunAbsY + fStepY),
                                                     -fLayerDistanceZ);
                        AddWire(surfaceContainer,
                                fGroupIndex,
                                fChDensities,
                                fWireConnectPA,
                                fWireConnectPB,
                                fWireDiameter,
                                fCathodeU,
                                1,
                                fWireConnectDisc);
                    }
                }
            }

            fRunAbsY += fStepY;
        }
    }

    if (!fUseWires) {

        if (fAnode) {
            fP0.SetComponents(-fWireLengthX, -fWireLengthX, fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fAnodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);

            fP0.SetComponents(0., -fWireLengthX, fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fAnodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);

            fP0.SetComponents(-fWireLengthX, 0., fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fAnodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);

            fP0.SetComponents(0., 0., fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fAnodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);
        }

        if (fCathode) {
            fP0.SetComponents(-fWireLengthX, -fWireLengthX, -fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fCathodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);

            fP0.SetComponents(0., -fWireLengthX, -fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fCathodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);

            fP0.SetComponents(-fWireLengthX, 0., -fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fCathodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);

            fP0.SetComponents(0., 0., -fLayerDistanceZ);
            AddRect(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fWireLengthX,
                    fWireLengthX,
                    fP0,
                    fNx,
                    fNy,
                    fCathodeU,
                    fRectElectrodeDisc,
                    1,
                    fRectElectrodeDisc);
        }
    }


    // ----------------------------------------------------------------------------

    if (fBoxXY) {
        // --------------------------
        // XY-layer at z=-fBoxHeightZ
        // --------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fXY269WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 2 */ fP0.SetComponents(-fBoxLengthX, -fXY269WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY269WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY26,
                fRectDiscXY26);

        /* No. 3 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 4 */ fP0.SetComponents(-fXY489LengthX / 2.e0, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY489LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY48,
                fRectDiscXY48);

        /* No. 5 */ fP0.SetComponents(fXY489LengthX / 2.e0, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 6 */ fP0.SetComponents(fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY269WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY26,
                fRectDiscXY26);

        /* No. 7 */ fP0.SetComponents(fXY489LengthX / 2.e0, fXY269WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 8 */ fP0.SetComponents(-fXY489LengthX / 2.e0, fXY269WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY489LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY48,
                fRectDiscXY48);

        /* No. 9 */ fP0.SetComponents(-fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY489LengthX,
                fXY269WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY9,
                fRectDiscXY9);

        // -------------------------
        // XY-layer at z=fBoxHeightZ
        // -------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fXY269WidthY / 2.e0, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 2 */ fP0.SetComponents(-fBoxLengthX, -fXY269WidthY / 2.e0, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY269WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY26,
                fRectDiscXY26);

        /* No. 3 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 4 */ fP0.SetComponents(-fXY489LengthX / 2.e0, -fBoxWidthY, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY489LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY48,
                fRectDiscXY48);

        /* No. 5 */ fP0.SetComponents(fXY489LengthX / 2.e0, -fBoxWidthY, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 6 */ fP0.SetComponents(fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY269WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY26,
                fRectDiscXY26);

        /* No. 7 */ fP0.SetComponents(fXY489LengthX / 2.e0, fXY269WidthY / 2.e0, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY123567LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY1357,
                fRectDiscXY1357);

        /* No. 8 */ fP0.SetComponents(-fXY489LengthX / 2.e0, fXY269WidthY / 2.e0, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY489LengthX,
                fXY134578WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY48,
                fRectDiscXY48);

        /* No. 9 */ fP0.SetComponents(-fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fXY489LengthX,
                fXY269WidthY,
                fP0,
                fNx,
                fNy,
                fBoxU,
                1,
                fRectDiscXY9,
                fRectDiscXY9);
    }

    // ----------------------------------------------------------------------------

    if (fBoundary) {
        // -----------------------------------------------
        // Virtual boundary XY-layer at z=fBoundaryHeightZ
        // -----------------------------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY123567LengthX,
                    fXY134578WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY1357,
                    fBoundaryDiscXY1357);

        /* No. 2 */ fP0.SetComponents(-fBoxLengthX, -fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY123567LengthX,
                    fXY269WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY26,
                    fBoundaryDiscXY26);

        /* No. 3 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY123567LengthX,
                    fXY134578WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY1357,
                    fBoundaryDiscXY1357);

        /* No. 4 */ fP0.SetComponents(-fXY489LengthX / 2.e0, -fBoxWidthY, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY489LengthX,
                    fXY134578WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY48,
                    fBoundaryDiscXY48);

        /* No. 5 */ fP0.SetComponents(fXY489LengthX / 2.e0, -fBoxWidthY, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY123567LengthX,
                    fXY134578WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY1357,
                    fBoundaryDiscXY1357);

        /* No. 6 */ fP0.SetComponents(fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY123567LengthX,
                    fXY269WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY26,
                    fBoundaryDiscXY26);

        /* No. 7 */ fP0.SetComponents(fXY489LengthX / 2.e0, fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY123567LengthX,
                    fXY134578WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY1357,
                    fBoundaryDiscXY1357);

        /* No. 8 */ fP0.SetComponents(-fXY489LengthX / 2.e0, fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY489LengthX,
                    fXY134578WidthY,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY48,
                    fBoundaryDiscXY48);

        /* No. 9 */ fP0.SetComponents(-fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, fBoundaryHeightZ);
        //      AddBoundary(surfaceContainer, fGroupIndex, fChDensities, fXY489LengthX, fXY269WidthY, fP0, fNx, fNy, fGXeEpsR, fLXeEpsR, 1,
        //              fBoundaryDiscXY9, fBoundaryDiscXY9);

        // Subdividing rectangle no. 9 into 4 additional rectangles:
        fP0.SetComponents(-fXY489LengthX / 2.e0, -fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY489LengthX / 2.,
                    fXY269WidthY / 2.,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY9,
                    fBoundaryDiscXY9);

        fP0.SetComponents(0., -fXY269WidthY / 2.e0, fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY489LengthX / 2.,
                    fXY269WidthY / 2.,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY9,
                    fBoundaryDiscXY9);

        fP0.SetComponents(-fXY489LengthX / 2.e0, 0., fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY489LengthX / 2.,
                    fXY269WidthY / 2.,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY9,
                    fBoundaryDiscXY9);

        fP0.SetComponents(0., 0., fBoundaryHeightZ);
        AddBoundary(surfaceContainer,
                    fGroupIndex,
                    fChDensities,
                    fXY489LengthX / 2.,
                    fXY269WidthY / 2.,
                    fP0,
                    fNx,
                    fNy,
                    fGXeEpsR,
                    fLXeEpsR,
                    1,
                    fBoundaryDiscXY9,
                    fBoundaryDiscXY9);
    }

    // ----------------------------------------------------------------------------

    if (fBoxYZ) {
        // -------------------------------
        // YZ-layer at x=-fBoxLengthX, z<0
        // -------------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fYZ25WidthY / 2.e0, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 2 */ fP0.SetComponents(-fBoxLengthX, -fYZ25WidthY / 2.e0, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ2,
                fRectDiscYZ2);

        /* No. 3 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 4 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        /* No. 5 */ fP0.SetComponents(-fBoxLengthX, -fYZ25WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ5,
                fRectDiscYZ5);

        /* No. 6 */ fP0.SetComponents(-fBoxLengthX, fYZ25WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        // ------------------------------
        // YZ-layer at x=fBoxLengthX, z<0
        // ------------------------------

        /* No. 1 */ fP0.SetComponents(fBoxLengthX, fYZ25WidthY / 2.e0, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 2 */ fP0.SetComponents(fBoxLengthX, -fYZ25WidthY / 2.e0, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ2,
                fRectDiscYZ2);

        /* No. 3 */ fP0.SetComponents(fBoxLengthX, -fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 4 */ fP0.SetComponents(fBoxLengthX, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        /* No. 5 */ fP0.SetComponents(fBoxLengthX, -fYZ25WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ5,
                fRectDiscYZ5);

        /* No. 6 */ fP0.SetComponents(fBoxLengthX, fYZ25WidthY / 2.e0, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        // -------------------------------
        // YZ-layer at x=-fBoxLengthX, z>0
        // -------------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fYZ25WidthY / 2.e0, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 2 */ fP0.SetComponents(-fBoxLengthX, -fYZ25WidthY / 2.e0, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ2,
                fRectDiscYZ2);

        /* No. 3 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 4 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        /* No. 5 */ fP0.SetComponents(-fBoxLengthX, -fYZ25WidthY / 2.e0, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ5,
                fRectDiscYZ5);

        /* No. 6 */ fP0.SetComponents(-fBoxLengthX, fYZ25WidthY / 2.e0, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        // ------------------------------
        // YZ-layer at x=fBoxLengthX, z>0
        // ------------------------------

        /* No. 1 */ fP0.SetComponents(fBoxLengthX, fYZ25WidthY / 2.e0, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 2 */ fP0.SetComponents(fBoxLengthX, -fYZ25WidthY / 2.e0, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ2,
                fRectDiscYZ2);

        /* No. 3 */ fP0.SetComponents(fBoxLengthX, -fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ123HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ13,
                fRectDiscYZ13);

        /* No. 4 */ fP0.SetComponents(fBoxLengthX, -fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);

        /* No. 5 */ fP0.SetComponents(fBoxLengthX, -fYZ25WidthY / 2.e0, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ25WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ5,
                fRectDiscYZ5);

        /* No. 6 */ fP0.SetComponents(fBoxLengthX, fYZ25WidthY / 2.e0, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fYZ1346WidthY,
                fYZ456HeightZ,
                fP0,
                fNy,
                fNz,
                fBoxU,
                1,
                fRectDiscYZ46,
                fRectDiscYZ46);
    }

    // ----------------------------------------------------------------------------

    if (fBoxZX) {
        // ------------------------------
        // ZX-layer at y=-fBoxWidthY, z<0
        // ------------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 2 */ fP0.SetComponents(-fZX25LengthX / 2.e0, -fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX2,
                fRectDiscZX2);

        /* No. 3 */ fP0.SetComponents(fZX25LengthX / 2.e0, -fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 4 */ fP0.SetComponents(fZX25LengthX / 2.e0, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        /* No. 5 */ fP0.SetComponents(-fZX25LengthX / 2.e0, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX5,
                fRectDiscZX5);

        /* No. 6 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        // -----------------------------
        // ZX-layer at y=fBoxWidthY, z<0
        // -----------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 2 */ fP0.SetComponents(-fZX25LengthX / 2.e0, fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX2,
                fRectDiscZX2);

        /* No. 3 */ fP0.SetComponents(fZX25LengthX / 2.e0, fBoxWidthY, -fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 4 */ fP0.SetComponents(fZX25LengthX / 2.e0, fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        /* No. 5 */ fP0.SetComponents(-fZX25LengthX / 2.e0, fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX5,
                fRectDiscZX5);

        /* No. 6 */ fP0.SetComponents(-fBoxLengthX, fBoxWidthY, -fBoxHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        // ------------------------------
        // ZX-layer at y=-fBoxWidthY, z>0
        // ------------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 2 */ fP0.SetComponents(-fZX25LengthX / 2.e0, -fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX2,
                fRectDiscZX2);

        /* No. 3 */ fP0.SetComponents(fZX25LengthX / 2.e0, -fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 4 */ fP0.SetComponents(fZX25LengthX / 2.e0, -fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        /* No. 5 */ fP0.SetComponents(-fZX25LengthX / 2.e0, -fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX5,
                fRectDiscZX5);

        /* No. 6 */ fP0.SetComponents(-fBoxLengthX, -fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        // -----------------------------
        // ZX-layer at y=fBoxWidthY, z>0
        // -----------------------------

        /* No. 1 */ fP0.SetComponents(-fBoxLengthX, fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 2 */ fP0.SetComponents(-fZX25LengthX / 2.e0, fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX2,
                fRectDiscZX2);

        /* No. 3 */ fP0.SetComponents(fZX25LengthX / 2.e0, fBoxWidthY, fBoundaryHeightZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX123HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX13,
                fRectDiscZX13);

        /* No. 4 */ fP0.SetComponents(fZX25LengthX / 2.e0, fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);

        /* No. 5 */ fP0.SetComponents(-fZX25LengthX / 2.e0, fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX25LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX5,
                fRectDiscZX5);

        /* No. 6 */ fP0.SetComponents(-fBoxLengthX, fBoxWidthY, fLayerDistanceZ);
        AddRect(surfaceContainer,
                fGroupIndex,
                fChDensities,
                fZX456HeightZ,
                fZX1346LengthX,
                fP0,
                fNz,
                fNx,
                fBoxU,
                1,
                fRectDiscZX46,
                fRectDiscZX46);
    }

    // ----------------------------------------------------------------------------

    KEBIPolicy fIntegratorPolicy;
#ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(surfaceContainer);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator{fIntegratorPolicy.CreateOpenCLIntegrator(*oclContainer)};
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(*oclContainer, integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(*oclContainer, integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(*oclContainer, integrator);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;

#else
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()}; /* default: numeric integrator */

    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    //    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
    //    gaussianElimination.Solve(A,x,b);

#ifdef MULTIRH
    KMultiElementRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
#else
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
#endif

#endif
    robinHood.SetTolerance(1.e-8);
    robinHood.SetResidualCheckInterval(1000);

    robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());

    auto* tracker = new KIterationTracker<KElectrostaticBoundaryIntegrator::ValueType>();
    tracker->Interval(1);
    tracker->WriteInterval(100);
    tracker->MaxIterationStamps(1.e6);
    robinHood.AddVisitor(tracker);
    auto* stateWriter = new KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>(surfaceContainer);
    stateWriter->Interval(50000);
    stateWriter->SaveNameRoot("TestDielectrics");
    robinHood.AddVisitor(stateWriter);

    robinHood.Solve(A, x, b);

    // ----------------------------------------------------------------------------

    clock_t fTimeChDenStart(0);
    clock_t fTimeChDenEnd(0);

    double fTimeChDen1(0.);
    KFieldVector fTimeChDen2(0., 0., 0.);

    fTimeChDenStart = clock();

#ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer* oclContainer2;
    oclContainer2 = new KOpenCLSurfaceContainer(surfaceContainer);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer2);
    KOpenCLElectrostaticBoundaryIntegrator* fOCLIntegrator2 =
        new KOpenCLElectrostaticBoundaryIntegrator{fIntegratorPolicy.CreateOpenCLConfig(), *oclContainer2};
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver =
        new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer2, *fOCLIntegrator2);
    direct_solver->Initialize();
#else
    auto* direct_solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer, integrator);
#endif


    fTimeChDenEnd = clock();

    fTimeChDen1 = ((double) (fTimeChDenEnd - fTimeChDenStart)) / CLOCKS_PER_SEC;
    fTimeChDen2 = CalcTime(fTimeChDen1);

    // ----------------------------------------------------------------------------

    std::cout << fGroupIndex << " wire and rectangle objects have been defined and discretized into" << std::endl;
    std::cout << fChDensities << " elements with independent charge densities." << std::endl << std::endl;
    std::cout << "Time for charge density calculation:" << std::endl;
    std::cout << fTimeChDen2.Z() << " hours , " << fTimeChDen2.Y() << " minutes , " << fTimeChDen2.X() << " seconds."
              << std::endl
              << std::endl;

    // ----------------------------------------------------------------------------

    clock_t fTimeFieldStart(0);
    clock_t fTimeFieldEnd(0);

    double fTimeField1(0.);
    KFieldVector fTimeField2(0., 0., 0.);

    fTimeFieldStart = clock();


    KFieldVector gEField(0., 0., 0.);
    double gPotential(0.);


    auto* gROOTFile = new TFile(fFieldOutput.c_str(), "RECREATE");
    auto* gGraphEField = new TGraph(gSteps);
    auto* gGraphPot = new TGraph(gSteps);
    auto* gGraphEFieldR = new TGraph(gSteps);
    auto* gGraphD = new TGraph(gSteps);

    TCanvas gCanvas("gCanvas", "Electrostatics of two-phase cube", 0, 0, 800, 600);
    gCanvas.SetTitle("TestDielectrics");
    gGraphEField->SetTitle("Electric Field in z-Direction;z [m];E_z [V/m]");
    gGraphPot->SetTitle("Electric Potential;z [m];U [V]");
    gGraphEFieldR->SetTitle("Electric Field in radial direction;z [m];E_r [V/m]");
    gGraphD->SetTitle("Electric displacement field in z-Direction;z [m];D_z [V/m]");

    double Er(0.);
    std::pair<KFieldVector, double> result;

    for (int i = 0; i < gSteps; i++) {

        result = direct_solver->ElectricFieldAndPotential(gLocation);
        gEField = result.first;
        gPotential = result.second;

        if (gLocation[2] < 0.2 && gLocation[2] > -0.2) {
            Er = sqrt((gEField[0] * gEField[0]) + (gEField[1] * gEField[1]));
            gGraphEField->SetPoint(i, gLocation[2], gEField[2]);
            gGraphPot->SetPoint(i, gLocation[2], gPotential);
            gGraphEFieldR->SetPoint(i, gLocation[2], Er);

            if (fBoundary) {
                if (gLocation[2] < 0) {
                    gGraphD->SetPoint(i, gLocation[2], gEField[2] * fLXeEpsR);
                }

                else {
                    gGraphD->SetPoint(i, gLocation[2], gEField[2] * fGXeEpsR);
                }
            }

            else {

                gGraphD->SetPoint(i, gLocation[2], gEField[2]);
            }
        }

        gLocation[2] = gLocation[2] + gStepsize;
    }


    gGraphEField->Draw("AP");
    gGraphEFieldR->Draw("AP");
    gGraphPot->Draw("AP");
    gGraphD->Draw("AP");

    gGraphEField->SetMarkerColor(kRed);
    gGraphEField->SetMarkerSize(0.5);
    gGraphEField->SetMarkerStyle(8);

    gGraphEFieldR->SetMarkerColor(kRed);
    gGraphEFieldR->SetMarkerSize(0.5);
    gGraphEFieldR->SetMarkerStyle(8);

    gGraphPot->SetMarkerColor(kRed);
    gGraphPot->SetMarkerSize(0.5);
    gGraphPot->SetMarkerStyle(8);

    gGraphD->SetMarkerColor(kRed);
    gGraphD->SetMarkerSize(0.5);
    gGraphD->SetMarkerStyle(8);

    gGraphEField->Write("EField_Z");
    gGraphEFieldR->Write("EField_R");
    gGraphPot->Write("Potential");
    gGraphD->Write("D_Z");


    fTimeFieldEnd = clock();
    fTimeField1 = ((double) (fTimeFieldEnd - fTimeFieldStart)) / CLOCKS_PER_SEC;
    fTimeField2 = CalcTime(fTimeField1);

    // ----------------------------------------------------------------------------

    std::cout << "Time for field and potential calculation:" << std::endl;
    std::cout << "(" << 4 * k << " points)" << std::endl;
    std::cout << fTimeField2.Z() << " hours , " << fTimeField2.Y() << " minutes , " << fTimeField2.X() << " seconds."
              << std::endl;

    KEMFileInterface::GetInstance()->Write(surfaceContainer, "surfaceContainer");
    gROOTFile->Close();

    surfaceContainer.clear();
    // ----------------------------------------------------------------------------

    return 0;
}
