/*
 * DielectricsTest.cc
 *
 *  Created on: 27 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestDielectrics.cc
 */

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KChargeDensitySolver.hh"
#include "KEMConstants.hh"
#include "KEMFieldTest.hh"
#include "KEMTicker.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KGaussianElimination.hh"
#include "KMultiElementRobinHood.hh"
#include "KRobinHood.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
#else
#include "KRobinHood_OpenCL.hh"
#endif
#endif

using namespace KEMField;

// typedefs for Dirichlet and Neumann elements
typedef KSurface<KElectrostaticBasis, KNeumannBoundary, KRectangle> KEMBoundary;
using KEMBoundaryTriangle = KSurface<KElectrostaticBasis, KNeumannBoundary, KTriangle>;
using KEMRectangle = KSurface<KElectrostaticBasis, KDirichletBoundary, KRectangle>;
using KEMWire = KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment>;

class KEMFieldDielectricsTest : public KEMFieldTest
{
  protected:
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

            for (int i = 0; i < nSegments / 2; i++) {
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

                // set length B
                rectangle->SetB(b[j]);
                dB += b[j];

                // add r to the surface container
                fContainer.push_back(rectangle);
            } /*B direction*/
            dA += a[i];
        } /*A direction*/

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

    void AddBoundary(KSurfaceContainer& fContainer, int& fGroup, int& fChDen, double fA, double fB,
                     const KFieldVector& fP0, const KFieldVector& fN1, const KFieldVector& fN2, double fEpsRAbove,
                     double fEpsRBelow, double fNRot, int fNumDiscA, int fNumDiscB)
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
                    std::cout << "The normal vectors N1 and N2 have been exchanged for "
                                 "positive N3.\n";
                    tri1->SetN1(fN2);
                    tri1->SetN2(fN1);
                    tri1->SetN3();
                }
                if (tri2->GetN3().Z() < 0) {
                    std::cout << "The normal vectors N1 and N2 have been exchanged for "
                                 "positive N3.\n";
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

  protected:
    void SetUp() override
    {
        KEMFieldTest::SetUp();

        // Technical definitions:
        std::string fModelName;
        std::string fMainDir;
        std::string ftextinput;
        int fGroupIndex(0);
        int fChDensities(0);
        int disc(4);

        // Physical input parameters:
        double fAnodeU(1500.);    // Voltage of anode (top)
        double fCathodeU(-500.);  // Voltage of cathode (bottom)
        double fGXeEpsR(1.);
        double fLXeEpsR(2.);

        // Geometrical setup:
        bool fAnode(true);
        bool fCathode(true);

        bool fBoundary(true);

        bool fUseWires(false);              // if true, then use parallel wires as electrodes, otherwise use plates
        bool fConnectWireEndPoints(false);  // Connect open end points of parallel anode and cathode wires

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

        // Discretization of the virtual boundary XY-layer:
        int fBoundaryDiscXY1357(disc);
        int fBoundaryDiscXY26(disc);
        int fBoundaryDiscXY48(disc);
        int fBoundaryDiscXY9(disc);

        // Discretization of electrode plates (XY-layer):
        double fRectElectrodeDisc(disc);

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
                        fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
                                                     fRunAbsY - fStepY,
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
                        fWireConnectPB.SetComponents(-fWireLengthX - fWireConnectDistX,
                                                     fRunAbsY - fStepY,
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
                        fWireConnectPB.SetComponents(fWireLengthX + fWireConnectDistX,
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
            //      AddBoundary(surfaceContainer, fGroupIndex, fChDensities,
            //      fXY489LengthX, fXY269WidthY, fP0, fNx, fNy, fGXeEpsR, fLXeEpsR, 1,
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

        MPI_SINGLE_PROCESS {
            std::cout << fGroupIndex << " wire and rectangle objects have been defined and discretized into" << std::endl;
            std::cout << fChDensities << " elements with independent charge densities." << std::endl;
        }
    }

    KSurfaceContainer surfaceContainer;
};

TEST_F(KEMFieldDielectricsTest, ElectrostaticBoundary)
{
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()}; /* default: numeric integrator */

    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KMultiElementRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;

    robinHood.SetTolerance(1.e-4);
    robinHood.SetResidualCheckInterval(1000);
    robinHood.Solve(A, x, b);

    auto* solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer, integrator);

    KFieldVector location;
    std::vector<double> locationZ = {-0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25};
    std::vector<double> phiRef = {-36.5, -69.9, -125.5, -218.1, -367.3, 123.3, 1092.5, 748.2, 522.2, 379.0, 286.7};

    for (unsigned i = 0; i < locationZ.size(); i++) {
        double phi;
        //KFieldVector E;

        location.SetZ(locationZ[i]);
        auto result = solver->ElectricFieldAndPotential(location);
        phi = result.second;
        //E = result.first;

        ASSERT_NEAR(phi, phiRef[i], 0.1);

        // NOTE: electric field is not tested
    }
}

/* FIXME - this test takes far too long when OpenCL is enabled (on CPU) */
//#if 0
#ifdef KEMFIELD_USE_OPENCL
TEST_F(KEMFieldDielectricsTest, ElectrostaticBoundary_OpenCL)
{
    KEBIPolicy integratorPolicy;

    KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(surfaceContainer);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
    KOpenCLElectrostaticBoundaryIntegrator* oclIntegrator =
        new KOpenCLElectrostaticBoundaryIntegrator{integratorPolicy.CreateOpenCLIntegrator(*oclContainer)};
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(*oclContainer, *oclIntegrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(*oclContainer, *oclIntegrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(*oclContainer, *oclIntegrator);

#ifdef KEMFIELD_USE_MPI
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL> robinHood;
#else
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
#endif  // KEMFIELD_USE_MPI

    robinHood.SetTolerance(1.e-4);
    robinHood.SetResidualCheckInterval(100);
    robinHood.Solve(A, x, b);

    KOpenCLSurfaceContainer* oclContainer2;
    oclContainer2 = new KOpenCLSurfaceContainer(surfaceContainer);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer2);
    KOpenCLElectrostaticBoundaryIntegrator* oclIntegrator2 =
        new KOpenCLElectrostaticBoundaryIntegrator{integratorPolicy.CreateOpenCLConfig(), *oclContainer2};
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* solver =
        new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer2, *oclIntegrator2);

    solver->Initialize();

    KFieldVector location;
    std::vector<double> locationZ = {-0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25};
    std::vector<double> phiRef = {-36.5, -69.9, -125.5, -218.1, -367.3, 123.3, 1092.5, 748.2, 522.2, 379.0, 286.7};

    for (unsigned i = 0; i < locationZ.size(); i++) {
        double phi;
        //KFieldVector E;

        location.SetZ(locationZ[i]);
        auto result = solver->ElectricFieldAndPotential(location);
        phi = result.second;
        //E = result.first;

        ASSERT_NEAR(phi, phiRef[i], 0.1);

        // NOTE: electric field is not tested
    }
}
#endif
//#endif
