#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"
#include "KSurfaceContainer.hh"
#include "KThreeVector_KEMField.hh"
#include "TApplication.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TLatex.h"
#include "TMultiGraph.h"
#include "TStyle.h"

#include <cstdlib>
#include <iostream>

#define POW2(x) ((x) * (x))

// VALUES
#define NUMLINES 1000    // number of line segments for each Dr step
#define MINDR    2       // minimal distance ratio to be investigated
#define MAXDR    10000   // maximal distance ratio to be investigated
#define STEPSDR  1000    // steps between given distance ratio range
#define ACCURACY 1.E-15  // targeted accuracy for both electric potential and field
#define SEPARATECOMP     // if this variable has been defined potentials and fields will be computed separately,
                         // hence 'ElectricFieldAndPotential' function won't be used
                         // both options have to produce same values
#define DRADDPERC 15     // additional fraction of distance ratio value at given accuracy to be added

// ROOT PLOTS AND COLORS (all settings apply for both field and potential)
#define PLOTANA    1
#define PLOTQUAD2  0
#define PLOTQUAD3  0
#define PLOTQUAD4  1
#define PLOTQUAD6  0
#define PLOTQUAD8  0
#define PLOTQUAD16 1
#define PLOTQUAD32 0
#define PLOTNUM    1

#define COLANA    kSpring
#define COLQUAD2  kAzure
#define COLQUAD3  kBlack
#define COLQUAD4  kRed
#define COLQUAD6  kCyan - 5
#define COLQUAD8  kPink
#define COLQUAD16 kOrange + 3
#define COLQUAD32 kMagenta
#define COLNUM    kBlue

#define LINEWIDTH 1.

using namespace KEMField;

double IJKLRANDOM;
typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment> KEMLineSegment;
void subrn(double* u, int len);
double randomnumber();

void printVec(const std::string& add, KFieldVector input)
{
    std::cout << add.c_str() << input.X() << "\t" << input.Y() << "\t" << input.Z() << std::endl;
}

namespace KEMField
{

// visitor for line segment geometry

class LineSegmentVisitor : public KSelectiveVisitor<KShapeVisitor, KTYPELIST_1(KLineSegment)>
{
  public:
    using KSelectiveVisitor<KShapeVisitor, KTYPELIST_1(KLineSegment)>::Visit;

    LineSegmentVisitor() = default;

    void Visit(KLineSegment& t) override
    {
        ProcessLineSegment(t);
    }

    void ProcessLineSegment(KLineSegment& l)
    {
        fLength = sqrt(POW2(l.GetP1()[0] - l.GetP0()[0]) + POW2(l.GetP1()[1] - l.GetP0()[1]) +
                       POW2(l.GetP1()[2] - l.GetP0()[2]));

        // centroid
        fShapeCentroid = l.Centroid();
    }

    double GetLength() const
    {
        return fLength;
    }
    KFieldVector GetCentroid()
    {
        return fShapeCentroid;
    }

  private:
    double fLength;
    KFieldVector fShapeCentroid;
};

// visitor for computing fields and potentials

class LineSegmentVisitorForElectricFieldAndPotential :
    public KSelectiveVisitor<KShapeVisitor, KTYPELIST_1(KLineSegment)>
{
  public:
    using KSelectiveVisitor<KShapeVisitor, KTYPELIST_1(KLineSegment)>::Visit;

    LineSegmentVisitorForElectricFieldAndPotential() = default;

    void Visit(KLineSegment& l) override
    {
        ComputeElectricFieldAndPotential(l);
    }

    void ComputeElectricFieldAndPotential(KLineSegment& l)
    {
        // line segment data in array form

        const double data[7] =
            {l.GetP0().X(), l.GetP0().Y(), l.GetP0().Z(), l.GetP1().X(), l.GetP1().Y(), l.GetP1().Z(), l.GetDiameter()};

#ifdef SEPARATECOMP
        // separate field and potential computation

        fQuad256ElectricFieldAndPotential = fQuad256Integrator.ElectricFieldAndPotential(&l, fP);

        fAnaElectricFieldAndPotential =
            std::make_pair(fAnaIntegrator.ElectricField(&l, fP), fAnaIntegrator.Potential(&l, fP));

        fQuad2ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 1, gQuadx2, gQuadw2),
                           fQuadIntegrator.Potential_nNodes(data, fP, 1, gQuadx2, gQuadw2));
        fQuad3ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 2, gQuadx3, gQuadw3),
                           fQuadIntegrator.Potential_nNodes(data, fP, 2, gQuadx3, gQuadw3));
        fQuad4ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 2, gQuadx4, gQuadw4),
                           fQuadIntegrator.Potential_nNodes(data, fP, 2, gQuadx4, gQuadw4));
        fQuad6ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 3, gQuadx6, gQuadw6),
                           fQuadIntegrator.Potential_nNodes(data, fP, 3, gQuadx6, gQuadw6));
        fQuad8ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 4, gQuadx8, gQuadw8),
                           fQuadIntegrator.Potential_nNodes(data, fP, 4, gQuadx8, gQuadw8));
        fQuad16ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 8, gQuadx16, gQuadw16),
                           fQuadIntegrator.Potential_nNodes(data, fP, 8, gQuadx16, gQuadw16));
        fQuad32ElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField_nNodes(data, fP, 16, gQuadx32, gQuadw32),
                           fQuadIntegrator.Potential_nNodes(data, fP, 16, gQuadx32, gQuadw32));

        fNumElectricFieldAndPotential =
            std::make_pair(fQuadIntegrator.ElectricField(&l, fP), fQuadIntegrator.Potential(&l, fP));
#else
        // simultaneous field and potential computation

        fQuad256ElectricFieldAndPotential = fQuad256Integrator.ElectricFieldAndPotential(&l, fP);

        fAnaElectricFieldAndPotential = fAnaIntegrator.ElectricFieldAndPotential(&l, fP);

        fQuad2ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 1, gQuadx2, gQuadw2);
        fQuad3ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 2, gQuadx3, gQuadw3);
        fQuad4ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 2, gQuadx4, gQuadw4);
        fQuad6ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 3, gQuadx6, gQuadw6);
        fQuad8ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 4, gQuadx8, gQuadw8);
        fQuad16ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 8, gQuadx16, gQuadw16);
        fQuad32ElectricFieldAndPotential =
            fQuadIntegrator.ElectricFieldAndPotential_nNodes(data, fP, 16, gQuadx32, gQuadw32);

        fNumElectricFieldAndPotential = fQuadIntegrator.ElectricFieldAndPotential(&l, fP);
#endif
    }

    void SetPosition(const KPosition& p) const
    {
        fP = p;
    }

    std::pair<KFieldVector, double>& GetQuad256ElectricFieldAndPotential() const
    {
        return fQuad256ElectricFieldAndPotential;
    }

    std::pair<KFieldVector, double>& GetAnaElectricFieldAndPotential() const
    {
        return fAnaElectricFieldAndPotential;
    }

    std::pair<KFieldVector, double>& GetQuad2ElectricFieldAndPotential() const
    {
        return fQuad2ElectricFieldAndPotential;
    }
    std::pair<KFieldVector, double>& GetQuad3ElectricFieldAndPotential() const
    {
        return fQuad3ElectricFieldAndPotential;
    }
    std::pair<KFieldVector, double>& GetQuad4ElectricFieldAndPotential() const
    {
        return fQuad4ElectricFieldAndPotential;
    }
    std::pair<KFieldVector, double>& GetQuad6ElectricFieldAndPotential() const
    {
        return fQuad6ElectricFieldAndPotential;
    }
    std::pair<KFieldVector, double>& GetQuad8ElectricFieldAndPotential() const
    {
        return fQuad8ElectricFieldAndPotential;
    }
    std::pair<KFieldVector, double>& GetQuad16ElectricFieldAndPotential() const
    {
        return fQuad16ElectricFieldAndPotential;
    }
    std::pair<KFieldVector, double>& GetQuad32ElectricFieldAndPotential() const
    {
        return fQuad32ElectricFieldAndPotential;
    }

    std::pair<KFieldVector, double>& GetNumElectricFieldAndPotential() const
    {
        return fNumElectricFieldAndPotential;
    }

  private:
    mutable KPosition fP;

    // 256-point quadrature as reference
    mutable std::pair<KFieldVector, double> fQuad256ElectricFieldAndPotential;
    KElectrostatic256NodeQuadratureLineSegmentIntegrator fQuad256Integrator;

    // analytical integration
    mutable std::pair<KFieldVector, double> fAnaElectricFieldAndPotential;
    KElectrostaticAnalyticLineSegmentIntegrator fAnaIntegrator;

    // quadrature n-node integration rules
    mutable std::pair<KFieldVector, double> fQuad2ElectricFieldAndPotential;
    mutable std::pair<KFieldVector, double> fQuad3ElectricFieldAndPotential;
    mutable std::pair<KFieldVector, double> fQuad4ElectricFieldAndPotential;
    mutable std::pair<KFieldVector, double> fQuad6ElectricFieldAndPotential;
    mutable std::pair<KFieldVector, double> fQuad8ElectricFieldAndPotential;
    mutable std::pair<KFieldVector, double> fQuad16ElectricFieldAndPotential;
    mutable std::pair<KFieldVector, double> fQuad32ElectricFieldAndPotential;
    KElectrostaticQuadratureLineSegmentIntegrator fQuadIntegrator;

    // adjusted quadrature integrator dependent from distance ratio
    mutable std::pair<KFieldVector, double> fNumElectricFieldAndPotential;
};

}  // namespace KEMField

int main()
{
    // This program determines the accuracy of the line segment integrators for a given distance ratio range.
    // distance ratio = distance to centroid / average side length

    // line segment data
    double P0[3];
    double P1[3];
    double length;

    // assign a unique direction vector for field point to each line segment and save into std::vector
    std::vector<KFieldVector> fPointDirections;

    // 'Num' line segments will be diced in the beginning and added to a surface container
    // This value decides how much 'line segments=field points' will be computed for each distance ratio value

    auto* container = new KSurfaceContainer();
    const unsigned int Num(NUMLINES); /* number of line segments */

    for (unsigned int i = 0; i < Num; i++) {
        IJKLRANDOM = i + 1;
        auto* line = new KEMLineSegment();

        // dice line segment geometry, diameter fixed ratio to length
        for (double& l : P0)
            l = -1. + 2. * randomnumber();
        for (double& j : P1)
            j = -1. + 2. * randomnumber();

        // compute further line segment data

        length = sqrt(POW2(P1[0] - P0[0]) + POW2(P1[1] - P0[1]) + POW2(P1[2] - P0[2]));

        line->SetP0(KFieldVector(P0[0], P0[1], P0[2]));
        line->SetP1(KFieldVector(P1[0], P1[1], P1[2]));
        line->SetDiameter(length * 0.1);

        line->SetBoundaryValue(1.);
        line->SetSolution(1.);

        container->push_back(line);

        const double costhetaFP = -1. + 2. * randomnumber();
        const double sinthetaFP = sqrt(1. - POW2(costhetaFP));
        const double phiFP = 2. * M_PI * randomnumber();

        fPointDirections.emplace_back(sinthetaFP * cos(phiFP), sinthetaFP * sin(phiFP), costhetaFP);
    }

    // visitor for elements
    LineSegmentVisitor fLineSegmentVisitor;
    LineSegmentVisitorForElectricFieldAndPotential fComputeVisitor;

    KSurfaceContainer::iterator it;

    // distance ratios
    const double minDr(MINDR);
    const double maxDr(MAXDR);
    double Dr = 0.;
    const unsigned int kmax(STEPSDR);
    const double C = log(maxDr / minDr) / kmax;

    KEMField::cout << "Iterate from dist. ratio " << minDr << " to " << maxDr << " in " << kmax << " steps."
                   << KEMField::endl;
    KEMField::cout << "Taking averaged relative error for " << container->size()
                   << " line segments for each dist. ratio value." << KEMField::endl;

    // field point
    KFieldVector fP;

    std::pair<KFieldVector, double> valQuad256;
    std::pair<KFieldVector, double> valAna;
    std::pair<KFieldVector, double> valQuad[7];
    std::pair<KFieldVector, double> valNum;

    // variables for accuracy check of n-node quadrature integration

    // potential
    bool accFlagPotQuad2(false);
    bool accFlagPotQuad3(false);
    bool accFlagPotQuad4(false);
    bool accFlagPotQuad6(false);
    bool accFlagPotQuad8(false);
    bool accFlagPotQuad16(false);
    bool accFlagPotQuad32(false);
    double drOptPotQuad2(0.);
    double drOptPotQuad3(0.);
    double drOptPotQuad4(0.);
    double drOptPotQuad6(0.);
    double drOptPotQuad8(0.);
    double drOptPotQuad16(0.);
    double drOptPotQuad32(0.);

    // field
    bool accFlagFieldQuad2(false);
    bool accFlagFieldQuad3(false);
    bool accFlagFieldQuad4(false);
    bool accFlagFieldQuad6(false);
    bool accFlagFieldQuad8(false);
    bool accFlagFieldQuad16(false);
    bool accFlagFieldQuad32(false);
    double drOptFieldQuad2(0.);
    double drOptFieldQuad3(0.);
    double drOptFieldQuad4(0.);
    double drOptFieldQuad6(0.);
    double drOptFieldQuad8(0.);
    double drOptFieldQuad16(0.);
    double drOptFieldQuad32(0.);

    // plot

    auto* fAppWindow = new TApplication("fAppWindow", nullptr, nullptr);

    gStyle->SetCanvasColor(kWhite);
    gStyle->SetLabelOffset(0.03, "xyz");  // values
    gStyle->SetTitleOffset(1.6, "xyz");   // label

    auto* mgPot = new TMultiGraph();

    auto* plotDrPotAna = new TGraph(kmax + 1);
    plotDrPotAna->SetTitle("Relative error of analytical line segment potential");
    plotDrPotAna->SetDrawOption("AC");
    plotDrPotAna->SetMarkerColor(COLANA);
    plotDrPotAna->SetLineWidth(LINEWIDTH);
    plotDrPotAna->SetLineColor(COLANA);
    plotDrPotAna->SetMarkerSize(0.2);
    plotDrPotAna->SetMarkerStyle(8);
    if (PLOTANA)
        mgPot->Add(plotDrPotAna);

    auto* plotDrPotQuad2 = new TGraph(kmax + 1);
    plotDrPotQuad2->SetTitle("Relative error of line segment 2-node quadrature potential");
    plotDrPotQuad2->SetDrawOption("same");
    plotDrPotQuad2->SetMarkerColor(COLQUAD2);
    plotDrPotQuad2->SetLineWidth(LINEWIDTH);
    plotDrPotQuad2->SetLineColor(COLQUAD2);
    plotDrPotQuad2->SetMarkerSize(0.2);
    plotDrPotQuad2->SetMarkerStyle(8);
    if (PLOTQUAD2)
        mgPot->Add(plotDrPotQuad2);

    auto* plotDrPotQuad3 = new TGraph(kmax + 1);
    plotDrPotQuad3->SetTitle("Relative error of line segment 3-node quadrature potential");
    plotDrPotQuad3->SetDrawOption("same");
    plotDrPotQuad3->SetMarkerColor(COLQUAD3);
    plotDrPotQuad3->SetLineWidth(LINEWIDTH);
    plotDrPotQuad3->SetLineColor(COLQUAD3);
    plotDrPotQuad3->SetMarkerSize(0.2);
    plotDrPotQuad3->SetMarkerStyle(8);
    if (PLOTQUAD3)
        mgPot->Add(plotDrPotQuad3);

    auto* plotDrPotQuad4 = new TGraph(kmax + 1);
    plotDrPotQuad4->SetTitle("Relative error of line segment 4-node quadrature potential");
    plotDrPotQuad4->SetDrawOption("same");
    plotDrPotQuad4->SetMarkerColor(COLQUAD4);
    plotDrPotQuad4->SetLineWidth(LINEWIDTH);
    plotDrPotQuad4->SetLineColor(COLQUAD4);
    plotDrPotQuad4->SetMarkerSize(0.2);
    plotDrPotQuad4->SetMarkerStyle(8);
    if (PLOTQUAD4)
        mgPot->Add(plotDrPotQuad4);

    auto* plotDrPotQuad6 = new TGraph(kmax + 1);
    plotDrPotQuad6->SetTitle("Relative error of line segment 6-node quadrature potential");
    plotDrPotQuad6->SetDrawOption("same");
    plotDrPotQuad6->SetMarkerColor(COLQUAD6);
    plotDrPotQuad6->SetLineWidth(LINEWIDTH);
    plotDrPotQuad6->SetLineColor(COLQUAD6);
    plotDrPotQuad6->SetMarkerSize(0.2);
    plotDrPotQuad6->SetMarkerStyle(8);
    if (PLOTQUAD6)
        mgPot->Add(plotDrPotQuad6);

    auto* plotDrPotQuad8 = new TGraph(kmax + 1);
    plotDrPotQuad8->SetTitle("Relative error of line segment 8-node quadrature potential");
    plotDrPotQuad8->SetDrawOption("same");
    plotDrPotQuad8->SetMarkerColor(COLQUAD8);
    plotDrPotQuad8->SetLineWidth(LINEWIDTH);
    plotDrPotQuad8->SetLineColor(COLQUAD8);
    plotDrPotQuad8->SetMarkerSize(0.2);
    plotDrPotQuad8->SetMarkerStyle(8);
    if (PLOTQUAD8)
        mgPot->Add(plotDrPotQuad8);

    auto* plotDrPotQuad16 = new TGraph(kmax + 1);
    plotDrPotQuad16->SetTitle("Relative error of line segment 16-node quadrature potential");
    plotDrPotQuad16->SetDrawOption("same");
    plotDrPotQuad16->SetMarkerColor(COLQUAD16);
    plotDrPotQuad16->SetLineWidth(LINEWIDTH);
    plotDrPotQuad16->SetLineColor(COLQUAD16);
    plotDrPotQuad16->SetMarkerSize(0.2);
    plotDrPotQuad16->SetMarkerStyle(8);
    if (PLOTQUAD16)
        mgPot->Add(plotDrPotQuad16);

    auto* plotDrPotQuad32 = new TGraph(kmax + 1);
    plotDrPotQuad32->SetTitle("Relative error of line segment 32-node quadrature potential");
    plotDrPotQuad32->SetDrawOption("same");
    plotDrPotQuad32->SetMarkerColor(COLQUAD32);
    plotDrPotQuad32->SetLineWidth(LINEWIDTH);
    plotDrPotQuad32->SetLineColor(COLQUAD32);
    plotDrPotQuad32->SetMarkerSize(0.2);
    plotDrPotQuad32->SetMarkerStyle(8);
    if (PLOTQUAD32)
        mgPot->Add(plotDrPotQuad32);

    auto* plotDrPotNum = new TGraph(kmax + 1);
    plotDrPotNum->SetTitle("Relative error of line segment potential with adjusted numerical integrator");
    plotDrPotNum->SetDrawOption("same");
    plotDrPotNum->SetMarkerColor(COLNUM);
    plotDrPotNum->SetLineWidth(LINEWIDTH);
    plotDrPotNum->SetLineColor(COLNUM);
    plotDrPotNum->SetMarkerSize(0.2);
    plotDrPotNum->SetMarkerStyle(8);
    if (PLOTNUM)
        mgPot->Add(plotDrPotNum);

    auto* mgField = new TMultiGraph();

    auto* plotDrFieldAna = new TGraph(kmax + 1);
    plotDrFieldAna->SetTitle("Relative error of analytical line segment field");
    plotDrFieldAna->SetDrawOption("AC");
    plotDrFieldAna->SetMarkerColor(COLANA);
    plotDrFieldAna->SetLineWidth(LINEWIDTH);
    plotDrFieldAna->SetLineColor(COLANA);
    plotDrFieldAna->SetMarkerSize(0.2);
    plotDrFieldAna->SetMarkerStyle(8);
    if (PLOTANA)
        mgField->Add(plotDrFieldAna);

    auto* plotDrFieldQuad2 = new TGraph(kmax + 1);
    plotDrFieldQuad2->SetTitle("Relative error of line segment 2-node quadrature field");
    plotDrFieldQuad2->SetDrawOption("same");
    plotDrFieldQuad2->SetMarkerColor(COLQUAD2);
    plotDrFieldQuad2->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad2->SetLineColor(COLQUAD2);
    plotDrFieldQuad2->SetMarkerSize(0.2);
    plotDrFieldQuad2->SetMarkerStyle(8);
    if (PLOTQUAD2)
        mgField->Add(plotDrFieldQuad2);

    auto* plotDrFieldQuad3 = new TGraph(kmax + 1);
    plotDrFieldQuad3->SetTitle("Relative error of line segment 3-node quadrature field");
    plotDrFieldQuad3->SetDrawOption("same");
    plotDrFieldQuad3->SetMarkerColor(COLQUAD3);
    plotDrFieldQuad3->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad3->SetLineColor(COLQUAD3);
    plotDrFieldQuad3->SetMarkerSize(0.2);
    plotDrFieldQuad3->SetMarkerStyle(8);
    if (PLOTQUAD3)
        mgField->Add(plotDrFieldQuad3);

    auto* plotDrFieldQuad4 = new TGraph(kmax + 1);
    plotDrFieldQuad4->SetTitle("Relative error of line segment 4-node quadrature field");
    plotDrFieldQuad4->SetDrawOption("same");
    plotDrFieldQuad4->SetMarkerColor(COLQUAD4);
    plotDrFieldQuad4->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad4->SetLineColor(COLQUAD4);
    plotDrFieldQuad4->SetMarkerSize(0.2);
    plotDrFieldQuad4->SetMarkerStyle(8);
    if (PLOTQUAD4)
        mgField->Add(plotDrFieldQuad4);

    auto* plotDrFieldQuad6 = new TGraph(kmax + 1);
    plotDrFieldQuad6->SetTitle("Relative error of line segment 6-node quadrature field");
    plotDrFieldQuad6->SetDrawOption("same");
    plotDrFieldQuad6->SetMarkerColor(COLQUAD6);
    plotDrFieldQuad6->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad6->SetLineColor(COLQUAD6);
    plotDrFieldQuad6->SetMarkerSize(0.2);
    plotDrFieldQuad6->SetMarkerStyle(8);
    if (PLOTQUAD6)
        mgField->Add(plotDrFieldQuad6);

    auto* plotDrFieldQuad8 = new TGraph(kmax + 1);
    plotDrFieldQuad8->SetTitle("Relative error of line segment 8-node quadrature field");
    plotDrFieldQuad8->SetDrawOption("same");
    plotDrFieldQuad8->SetMarkerColor(COLQUAD8);
    plotDrFieldQuad8->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad8->SetLineColor(COLQUAD8);
    plotDrFieldQuad8->SetMarkerSize(0.2);
    plotDrFieldQuad8->SetMarkerStyle(8);
    if (PLOTQUAD8)
        mgField->Add(plotDrFieldQuad8);

    auto* plotDrFieldQuad16 = new TGraph(kmax + 1);
    plotDrFieldQuad16->SetTitle("Relative error of line segment 16-node quadrature field");
    plotDrFieldQuad16->SetDrawOption("same");
    plotDrFieldQuad16->SetMarkerColor(COLQUAD16);
    plotDrFieldQuad16->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad16->SetLineColor(COLQUAD16);
    plotDrFieldQuad16->SetMarkerSize(0.2);
    plotDrFieldQuad16->SetMarkerStyle(8);
    if (PLOTQUAD16)
        mgField->Add(plotDrFieldQuad16);

    auto* plotDrFieldQuad32 = new TGraph(kmax + 1);
    plotDrFieldQuad32->SetTitle("Relative error of line segment 32-node quadrature field");
    plotDrFieldQuad32->SetDrawOption("same");
    plotDrFieldQuad32->SetMarkerColor(COLQUAD32);
    plotDrFieldQuad32->SetLineWidth(LINEWIDTH);
    plotDrFieldQuad32->SetLineColor(COLQUAD32);
    plotDrFieldQuad32->SetMarkerSize(0.2);
    plotDrFieldQuad32->SetMarkerStyle(8);
    if (PLOTQUAD32)
        mgField->Add(plotDrFieldQuad32);

    auto* plotDrFieldNum = new TGraph(kmax + 1);
    plotDrFieldNum->SetTitle("Relative error of triangle field with adjusted numerical integrator");
    plotDrFieldNum->SetDrawOption("same");
    plotDrFieldNum->SetMarkerColor(COLNUM);
    plotDrFieldNum->SetLineWidth(LINEWIDTH);
    plotDrFieldNum->SetLineColor(COLNUM);
    plotDrFieldNum->SetMarkerSize(0.2);
    plotDrFieldNum->SetMarkerStyle(8);
    if (PLOTNUM)
        mgField->Add(plotDrFieldNum);

    double relAnaPot(0.);
    double relQuad2Pot(0.);
    double relQuad3Pot(0.);
    double relQuad4Pot(0.);
    double relQuad6Pot(0.);
    double relQuad8Pot(0.);
    double relQuad16Pot(0.);
    double relQuad32Pot(0.);
    double relNumPot(0.);

    double relAnaField(0.);
    double relQuad2Field(0.);
    double relQuad3Field(0.);
    double relQuad4Field(0.);
    double relQuad6Field(0.);
    double relQuad8Field(0.);
    double relQuad16Field(0.);
    double relQuad32Field(0.);
    double relNumField(0.);

    const double targetAccuracy(ACCURACY);

    // iterate over distance ratios in log steps
    for (unsigned int k = 0; k <= kmax; k++) {

        Dr = minDr * exp(C * k);

        KEMField::cout << "Current distance ratio: " << Dr << "\t\r";
        KEMField::cout.flush();

        unsigned int directionIndex(0);

        // iterate over container elements
        for (it = container->begin<KElectrostaticBasis>(); it != container->end<KElectrostaticBasis>(); ++it) {

            IJKLRANDOM++;

            (*it)->Accept(fLineSegmentVisitor);

            // assign field point value
            fP = fLineSegmentVisitor.GetCentroid() +
                 fLineSegmentVisitor.GetLength() * Dr * fPointDirections[directionIndex];

            directionIndex++;

            fComputeVisitor.SetPosition(fP);

            (*it)->Accept(fComputeVisitor);

            valQuad256 = fComputeVisitor.GetQuad256ElectricFieldAndPotential();
            valAna = fComputeVisitor.GetAnaElectricFieldAndPotential();
            valQuad[0] = fComputeVisitor.GetQuad2ElectricFieldAndPotential();
            valQuad[1] = fComputeVisitor.GetQuad3ElectricFieldAndPotential();
            valQuad[2] = fComputeVisitor.GetQuad4ElectricFieldAndPotential();
            valQuad[3] = fComputeVisitor.GetQuad6ElectricFieldAndPotential();
            valQuad[4] = fComputeVisitor.GetQuad8ElectricFieldAndPotential();
            valQuad[5] = fComputeVisitor.GetQuad16ElectricFieldAndPotential();
            valQuad[6] = fComputeVisitor.GetQuad32ElectricFieldAndPotential();
            valNum = fComputeVisitor.GetNumElectricFieldAndPotential();

            // sum for relative error
            relAnaPot += fabs((valAna.second - valQuad256.second) / valQuad256.second);
            relQuad2Pot += fabs((valQuad[0].second - valQuad256.second) / valQuad256.second);
            relQuad3Pot += fabs((valQuad[1].second - valQuad256.second) / valQuad256.second);
            relQuad4Pot += fabs((valQuad[2].second - valQuad256.second) / valQuad256.second);
            relQuad6Pot += fabs((valQuad[3].second - valQuad256.second) / valQuad256.second);
            relQuad8Pot += fabs((valQuad[4].second - valQuad256.second) / valQuad256.second);
            relQuad16Pot += fabs((valQuad[5].second - valQuad256.second) / valQuad256.second);
            relQuad32Pot += fabs((valQuad[6].second - valQuad256.second) / valQuad256.second);
            relNumPot += fabs((valNum.second - valQuad256.second) / valQuad256.second);

            const double mag = sqrt(POW2(valQuad256.first[0]) + POW2(valQuad256.first[1]) + POW2(valQuad256.first[2]));

            for (unsigned short i = 0; i < 3; i++) {
                relAnaField += fabs(valAna.first[i] - valQuad256.first[i]) / (mag);
                relQuad2Field += fabs(valQuad[0].first[i] - valQuad256.first[i]) / (mag);
                relQuad3Field += fabs(valQuad[1].first[i] - valQuad256.first[i]) / (mag);
                relQuad4Field += fabs(valQuad[2].first[i] - valQuad256.first[i]) / (mag);
                relQuad6Field += fabs(valQuad[3].first[i] - valQuad256.first[i]) / (mag);
                relQuad8Field += fabs(valQuad[4].first[i] - valQuad256.first[i]) / (mag);
                relQuad16Field += fabs(valQuad[5].first[i] - valQuad256.first[i]) / (mag);
                relQuad32Field += fabs(valQuad[6].first[i] - valQuad256.first[i]) / (mag);
                relNumField += fabs(valNum.first[i] - valQuad256.first[i]) / (mag);
            }
        }

        relAnaPot /= Num;
        relQuad2Pot /= Num;
        relQuad3Pot /= Num;
        relQuad4Pot /= Num;
        relQuad6Pot /= Num;
        relQuad8Pot /= Num;
        relQuad16Pot /= Num;
        relQuad32Pot /= Num;
        relNumPot /= Num;

        relAnaField /= Num;
        relQuad2Field /= Num;
        relQuad3Field /= Num;
        relQuad4Field /= Num;
        relQuad6Field /= Num;
        relQuad8Field /= Num;
        relQuad16Field /= Num;
        relQuad32Field /= Num;
        relNumField /= Num;

        // potential

        if ((!accFlagPotQuad2) && (relQuad2Pot <= targetAccuracy)) {
            drOptPotQuad2 = Dr;
            accFlagPotQuad2 = true;
        }
        if ((!accFlagPotQuad3) && (relQuad3Pot <= targetAccuracy)) {
            drOptPotQuad3 = Dr;
            accFlagPotQuad3 = true;
        }
        if ((!accFlagPotQuad4) && (relQuad4Pot <= targetAccuracy)) {
            drOptPotQuad4 = Dr;
            accFlagPotQuad4 = true;
        }
        if ((!accFlagPotQuad6) && (relQuad6Pot <= targetAccuracy)) {
            drOptPotQuad6 = Dr;
            accFlagPotQuad6 = true;
        }
        if ((!accFlagPotQuad8) && (relQuad8Pot <= targetAccuracy)) {
            drOptPotQuad8 = Dr;
            accFlagPotQuad8 = true;
        }
        if ((!accFlagPotQuad16) && (relQuad16Pot <= targetAccuracy)) {
            drOptPotQuad16 = Dr;
            accFlagPotQuad16 = true;
        }
        if ((!accFlagPotQuad32) && (relQuad32Pot <= relAnaPot)) {
            drOptPotQuad32 = Dr;
            accFlagPotQuad32 = true;
        }

        // field

        if ((!accFlagFieldQuad2) && (relQuad2Field <= targetAccuracy)) {
            drOptFieldQuad2 = Dr;
            accFlagFieldQuad2 = true;
        }
        if ((!accFlagFieldQuad3) && (relQuad3Field <= targetAccuracy)) {
            drOptFieldQuad3 = Dr;
            accFlagFieldQuad3 = true;
        }
        if ((!accFlagFieldQuad4) && (relQuad4Field <= targetAccuracy)) {
            drOptFieldQuad4 = Dr;
            accFlagFieldQuad4 = true;
        }
        if ((!accFlagFieldQuad6) && (relQuad6Field <= targetAccuracy)) {
            drOptFieldQuad6 = Dr;
            accFlagFieldQuad6 = true;
        }
        if ((!accFlagFieldQuad8) && (relQuad8Field <= targetAccuracy)) {
            drOptFieldQuad8 = Dr;
            accFlagFieldQuad8 = true;
        }
        if ((!accFlagFieldQuad16) && (relQuad16Field <= targetAccuracy)) {
            drOptFieldQuad16 = Dr;
            accFlagFieldQuad16 = true;
        }
        if ((!accFlagFieldQuad32) && (relQuad32Field <= relAnaField)) {
            drOptFieldQuad32 = Dr;
            accFlagFieldQuad32 = true;
        }

        // save relative error of each integrator
        if (PLOTANA)
            plotDrPotAna->SetPoint(k, Dr, relAnaPot);
        if (PLOTQUAD2)
            plotDrPotQuad2->SetPoint(k, Dr, relQuad2Pot);
        if (PLOTQUAD3)
            plotDrPotQuad3->SetPoint(k, Dr, relQuad3Pot);
        if (PLOTQUAD4)
            plotDrPotQuad4->SetPoint(k, Dr, relQuad4Pot);
        if (PLOTQUAD6)
            plotDrPotQuad6->SetPoint(k, Dr, relQuad6Pot);
        if (PLOTQUAD8)
            plotDrPotQuad8->SetPoint(k, Dr, relQuad8Pot);
        if (PLOTQUAD16)
            plotDrPotQuad16->SetPoint(k, Dr, relQuad16Pot);
        if (PLOTQUAD32)
            plotDrPotQuad32->SetPoint(k, Dr, relQuad32Pot);
        if (PLOTNUM)
            plotDrPotNum->SetPoint(k, Dr, relNumPot);

        // reset relative error
        relAnaPot = 0.;
        relQuad2Pot = 0.;
        relQuad3Pot = 0.;
        relQuad4Pot = 0.;
        relQuad6Pot = 0.;
        relQuad8Pot = 0.;
        relQuad16Pot = 0.;
        relQuad32Pot = 0.;
        relNumPot = 0.;

        if (PLOTANA)
            plotDrFieldAna->SetPoint(k, Dr, relAnaField);
        if (PLOTQUAD2)
            plotDrFieldQuad2->SetPoint(k, Dr, relQuad2Field);
        if (PLOTQUAD3)
            plotDrFieldQuad3->SetPoint(k, Dr, relQuad3Field);
        if (PLOTQUAD4)
            plotDrFieldQuad4->SetPoint(k, Dr, relQuad4Field);
        if (PLOTQUAD6)
            plotDrFieldQuad6->SetPoint(k, Dr, relQuad6Field);
        if (PLOTQUAD8)
            plotDrFieldQuad8->SetPoint(k, Dr, relQuad8Field);
        if (PLOTQUAD16)
            plotDrFieldQuad16->SetPoint(k, Dr, relQuad16Field);
        if (PLOTQUAD32)
            plotDrFieldQuad32->SetPoint(k, Dr, relQuad32Field);
        if (PLOTNUM)
            plotDrFieldNum->SetPoint(k, Dr, relNumField);

        relAnaField = 0.;
        relQuad2Field = 0.;
        relQuad3Field = 0.;
        relQuad4Field = 0.;
        relQuad6Field = 0.;
        relQuad8Field = 0.;
        relQuad16Field = 0.;
        relQuad32Field = 0.;
        relNumField = 0.;
    } /* distance ratio */

    const double drAdd(DRADDPERC / 100.);

    KEMField::cout << "Recommended distance ratio values for target accuracy " << targetAccuracy << " (+"
                   << (100 * drAdd) << "%):" << KEMField::endl;
    KEMField::cout << "Line segment potentials:" << KEMField::endl;
    KEMField::cout << "*  2-point quadrature: " << ((1. + drAdd) * drOptPotQuad2) << KEMField::endl;
    KEMField::cout << "*  3-point quadrature: " << ((1. + drAdd) * drOptPotQuad3) << KEMField::endl;
    KEMField::cout << "*  4-point quadrature: " << ((1. + drAdd) * drOptPotQuad4) << KEMField::endl;
    KEMField::cout << "*  6-point quadrature: " << ((1. + drAdd) * drOptPotQuad6) << KEMField::endl;
    KEMField::cout << "*  8-point quadrature: " << ((1. + drAdd) * drOptPotQuad8) << KEMField::endl;
    KEMField::cout << "* 16-point quadrature: " << ((1.) * drOptPotQuad16) << " (no tolerance set here)"
                   << KEMField::endl;
    KEMField::cout << "* 32-point quadrature: " << ((1.) * drOptPotQuad32) << " (no tolerance set here)"
                   << KEMField::endl;
    KEMField::cout << "Line segment fields:" << KEMField::endl;
    KEMField::cout << "*  2-point quadrature: " << ((1. + drAdd) * drOptFieldQuad2) << KEMField::endl;
    KEMField::cout << "*  3-point quadrature: " << ((1. + drAdd) * drOptFieldQuad3) << KEMField::endl;
    KEMField::cout << "*  4-point quadrature: " << ((1. + drAdd) * drOptFieldQuad4) << KEMField::endl;
    KEMField::cout << "*  6-point quadrature: " << ((1. + drAdd) * drOptFieldQuad6) << KEMField::endl;
    KEMField::cout << "*  8-point quadrature: " << ((1. + drAdd) * drOptFieldQuad8) << KEMField::endl;
    KEMField::cout << "* 16-point quadrature: " << ((1.) * drOptFieldQuad16) << " (no tolerance set here)"
                   << KEMField::endl;
    KEMField::cout << "* 32-point quadrature: " << ((1.) * drOptFieldQuad32) << " (no tolerance set here)"
                   << KEMField::endl;

    KEMField::cout << "Distance ratio analysis for quadrature integrators finished." << KEMField::endl;

    TCanvas cPot("cPot", "Averaged relative error of line segment potential", 0, 0, 960, 760);
    cPot.SetMargin(0.16, 0.06, 0.15, 0.06);
    cPot.SetLogx();
    cPot.SetLogy();

    // multigraph, create plot
    cPot.cd();

    mgPot->Draw("apl");
    mgPot->SetTitle("Averaged error of line segment potential");
    mgPot->GetXaxis()->SetTitle("distance ratio");
    mgPot->GetXaxis()->CenterTitle();
    mgPot->GetYaxis()->SetTitle("relative error");
    mgPot->GetYaxis()->CenterTitle();

    TLatex l;
    l.SetTextAlign(11);
    l.SetTextFont(62);
    l.SetTextSize(0.032);

    if (PLOTANA) {
        l.SetTextAngle(29);
        l.SetTextColor(COLANA);
        l.DrawLatex(500, 1.5e-9, "Analytical");
    }

    if (PLOTQUAD2) {
        l.SetTextAngle(-43);
        l.SetTextColor(COLQUAD2);
        l.DrawLatex(9, 1.e-6, "2-node quadrature");
    }

    if (PLOTQUAD3) {
        l.SetTextAngle(-54);
        l.SetTextColor(COLQUAD3);
        l.DrawLatex(4., 1.e-7, "3-node quadrature");
    }

    if (PLOTQUAD4) {
        l.SetTextAngle(-54);
        l.SetTextColor(COLQUAD4);
        l.DrawLatex(4., 1.e-7, "4-node quadrature");
    }

    if (PLOTQUAD6) {
        l.SetTextAngle(-65);
        l.SetTextColor(COLQUAD6);
        l.DrawLatex(2.9, 1.e-8, "6-node quadrature");
    }

    if (PLOTQUAD8) {
        l.SetTextAngle(-69);
        l.SetTextColor(COLQUAD8);
        l.DrawLatex(2.5, 3e-10, "8-node quadrature");
    }

    if (PLOTQUAD16) {
        l.SetTextAngle(-69);
        l.SetTextColor(COLQUAD16);
        l.DrawLatex(2.5, 3e-10, "16-node quadrature");
    }

    if (PLOTQUAD32) {
        l.SetTextAngle(0);
        l.SetTextColor(COLQUAD32);
        l.DrawLatex(3., 2.e-16, "32-point quadrature");
    }

    if (PLOTNUM) {
        l.SetTextAngle(-69);
        l.SetTextColor(COLNUM);
        l.DrawLatex(2.5, 3e-10, "Numerical quadrature + analytical");
    }

    cPot.Update();

    TCanvas cField("cField", "Averaged relative error of line segment field", 0, 0, 960, 760);
    cField.SetMargin(0.16, 0.06, 0.15, 0.06);
    cField.SetLogx();
    cField.SetLogy();

    // multigraph, create plot
    cField.cd();
    mgField->Draw("apl");
    mgField->SetTitle("Averaged error of line segment field");
    mgField->GetXaxis()->SetTitle("distance ratio");
    mgField->GetXaxis()->CenterTitle();
    mgField->GetYaxis()->SetTitle("relative error");
    mgField->GetYaxis()->CenterTitle();

    if (PLOTANA) {
        l.SetTextAngle(29);
        l.SetTextColor(COLANA);
        l.DrawLatex(500, 1.5e-9, "Analytical");
    }

    if (PLOTQUAD2) {
        l.SetTextAngle(-43);
        l.SetTextColor(COLQUAD2);
        l.DrawLatex(9, 1.e-6, "2-node quadrature");
    }

    if (PLOTQUAD3) {
        l.SetTextAngle(-54);
        l.SetTextColor(COLQUAD3);
        l.DrawLatex(4., 1.e-7, "3-node quadrature");
    }

    if (PLOTQUAD4) {
        l.SetTextAngle(-54);
        l.SetTextColor(COLQUAD4);
        l.DrawLatex(4., 1.e-7, "4-node quadrature");
    }

    if (PLOTQUAD6) {
        l.SetTextAngle(-65);
        l.SetTextColor(COLQUAD6);
        l.DrawLatex(2.9, 1.e-8, "6-node quadrature");
    }

    if (PLOTQUAD8) {
        l.SetTextAngle(-69);
        l.SetTextColor(COLQUAD8);
        l.DrawLatex(2.5, 3e-10, "8-node quadrature");
    }

    if (PLOTQUAD16) {
        l.SetTextAngle(-69);
        l.SetTextColor(COLQUAD16);
        l.DrawLatex(2.5, 3e-10, "16-node quadrature");
    }

    if (PLOTQUAD32) {
        l.SetTextAngle(0);
        l.SetTextColor(COLQUAD32);
        l.DrawLatex(3., 2.e-16, "32-point quadrature");
    }

    if (PLOTNUM) {
        l.SetTextAngle(-69);
        l.SetTextColor(COLNUM);
        l.DrawLatex(2.5, 3e-10, "Numerical quadrature + analytical");
    }

    cField.Update();

    fAppWindow->Run();

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
