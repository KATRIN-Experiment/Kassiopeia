#include "KSROOTZonalHarmonicsPainter.h"

#include "KElectricZHFieldSolver.hh"
#include "KGElectrostaticBoundaryField.hh"
#include "KGStaticElectromagnetField.hh"
#include "KSObject.h"
#include "KSFieldFinder.h"
#include "KSElectricKEMField.h"
#include "KSMagneticKEMField.h"
#include "KSVisualizationMessage.h"
#include "KToolbox.h"
#include "KZonalHarmonicMagnetostaticFieldSolver.hh"

#include "TMultiGraph.h"
#include "TGraph.h"
#include "TEllipse.h"
#include "TMarker.h"

#include <fstream>
#include <iostream>
#include <limits>

using namespace KEMField;
using namespace KGeoBag;
using namespace katrin;
using namespace std;

namespace Kassiopeia
{
KSROOTZonalHarmonicsPainter::KSROOTZonalHarmonicsPainter() :
    fXAxis("z"),
    fYAxis("r"),
    fZMin(0.0),
    fZMax(0.0),
    fRMin(0.0),
    fRMax(5.0),
    fZDist(0.1),
    fRDist(0.01),
    fZMaxSteps(5000),
    fRMaxSteps(5000),
    fElectricFieldName(""),
    fMagneticFieldName(""),
    fFile(""),
    fPath(""),
    fDrawConvergenceArea(true),
    fDrawSourcePoints(false),
    fDrawCentralBoundary(false),
    fDrawRemoteBoundary(false)
{}
KSROOTZonalHarmonicsPainter::~KSROOTZonalHarmonicsPainter() = default;

void KSROOTZonalHarmonicsPainter::Render()
{
    if (fZMin >= fZMax) {
        fZMin = GetWindow()->GetXMin();
        fZMax = GetWindow()->GetXMax();
    }
    if (fRMin >= fRMax) {
        fRMin = GetWindow()->GetYMin() > 0. ? GetWindow()->GetYMin() : 0.;
        fRMax = GetWindow()->GetYMax();
    }

    bool autoAdjustZ = (fZMin >= fZMax);
    if (autoAdjustZ) {
        fZMin = std::numeric_limits<double>::max();
        fZMax = std::numeric_limits<double>::min();
    }

    fElCentralConvergenceBounds.clear();
    fElRemoteConvergenceBounds.clear();
    fElCentralSourcePoints.clear();
    fElRemoteSourcePoints.clear();

    fMagCentralConvergenceBounds.clear();
    fMagRemoteConvergenceBounds.clear();
    fMagCentralSourcePoints.clear();
    fMagRemoteSourcePoints.clear();

    KElectricZHFieldSolver* tElZHSolver = nullptr;
    if (!fElectricFieldName.empty()) {
        vismsg(eNormal) << "Getting electric field " << fElectricFieldName << " from the toolbox" << eom;

        auto* tElField = katrin::KToolbox::GetInstance().Get<KGElectrostaticBoundaryField>(fElectricFieldName);
        if (! tElField) {
            KSElectricField* tFieldWrapper = getElectricField(fElectricFieldName);
            auto* tKEMFieldObject = dynamic_cast<KSElectricKEMField*>(tFieldWrapper);
            tElField = dynamic_cast<KGElectrostaticBoundaryField*>(tKEMFieldObject->GetElectricField());
        }

        if (tElField == nullptr)
            vismsg(eError) << "No electric Field!" << eom;
        vismsg(eNormal) << "Initialize electric field (again)" << eom;
        tElField->Initialize();

        //vismsg(eNormal) << "retrieve converter from electric field" << eom;
        auto tElConverter = tElField->GetConverter();
        if (! tElConverter)
            vismsg(eError) << "Electric Field has no Converter! " << eom;

        //vismsg(eNormal) << "retrieve ZH solver from electric field" << eom;
        tElZHSolver = dynamic_cast<KElectricZHFieldSolver*>(&(*tElField->GetFieldSolver()));
        if (tElZHSolver == nullptr)
            vismsg(eError) << "Electric Field has no ZHSolver!" << eom;

        for (const auto& sp : tElZHSolver->GetContainer()->CentralSourcePoints()) {
            if (autoAdjustZ) {
                if (sp.first < fZMin)
                    fZMin = sp.first;
                if (sp.first > fZMax)
                    fZMax = sp.first;
            }
            fElCentralSourcePoints.push_back({sp.first, sp.second});
        }
        for (const auto& sp : tElZHSolver->GetContainer()->RemoteSourcePoints()) {
            if (autoAdjustZ) {
                if (sp.first < fZMin)
                    fZMin = sp.first;
                if (sp.first > fZMax)
                    fZMax = sp.first;
            }
            fElRemoteSourcePoints.push_back({sp.first, sp.second});
        }
    }

    KZonalHarmonicMagnetostaticFieldSolver* tMagZHSolver = nullptr;
    if (!fMagneticFieldName.empty()) {
        vismsg(eNormal) << "Getting magnetic field " << fMagneticFieldName << " from the toolbox" << eom;

        auto* tMagField = katrin::KToolbox::GetInstance().Get<KGStaticElectromagnetField>(fMagneticFieldName);
        if (! tMagField) {
            KSMagneticField* tFieldWrapper = getMagneticField(fMagneticFieldName);
            auto* tKEMFieldObject = dynamic_cast<KSMagneticKEMField*>(tFieldWrapper);
            tMagField = dynamic_cast<KGStaticElectromagnetField*>(tKEMFieldObject->GetMagneticField());
        }

        if (tMagField == nullptr)
            vismsg(eError) << "No magnetic Field!" << eom;
        vismsg(eNormal) << "Initialize magnetic field (again)" << eom;
        tMagField->Initialize();

        //vismsg(eNormal) << "retrieve ZH solver from magnetic field" << eom;
        tMagZHSolver = dynamic_cast<KZonalHarmonicMagnetostaticFieldSolver*>(&(*tMagField->GetFieldSolver()));
        if (tMagZHSolver == nullptr)
            vismsg(eError) << "Magnetic Field has no ZHSolver!" << eom;

        for (const auto& sp : tMagZHSolver->GetContainer()->CentralSourcePoints()) {
            if (autoAdjustZ) {
                if (sp.first < fZMin)
                    fZMin = sp.first;
                if (sp.first > fZMax)
                    fZMax = sp.first;
            }
            fMagCentralSourcePoints.push_back({sp.first, sp.second});
        }
        for (const auto& sp : tMagZHSolver->GetContainer()->RemoteSourcePoints()) {
            if (autoAdjustZ) {
                if (sp.first < fZMin)
                    fZMin = sp.first;
                if (sp.first > fZMax)
                    fZMax = sp.first;
            }
            fMagRemoteSourcePoints.push_back({sp.first, sp.second});
        }
    }

    double tDeltaZ = fZDist;
    if (ceil(fabs(fZMax - fZMin) / tDeltaZ) > fZMaxSteps) {
        tDeltaZ = fabs(fZMax - fZMin) / fZMaxSteps;
    }
    double tDeltaR = fRDist;
    if (ceil(fabs(fRMax - fRMin) / tDeltaR) > fRMaxSteps) {
        tDeltaR = fabs(fRMax - fRMin) / fRMaxSteps;
    }

    unsigned tNPoints = floor((fZMax - fZMin) / fZDist);
    vismsg(eNormal) << "ZH painter: start calculating convergence boundary from " << fZMin << " to " << fZMax << " m ("
                    << tNPoints << " steps) ..." << eom;

    for (unsigned tPointIndex = 0; tPointIndex < tNPoints; tPointIndex++) {
        //unsigned tNegPointIndex = 2*tNPoints - tPointIndex - 1;
        double tZ = fZMin + tPointIndex * tDeltaZ;

        if (tElZHSolver != nullptr) {
            double tR = 0;
            for (const auto& sp : tElZHSolver->GetContainer()->CentralSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                if (dz > rho)
                    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h > tR)
                    tR = h;
            }

            vismsg_debug("  electric field central radius at z=" << tZ << " is r=" << tR << " m" << eom);
            fElCentralConvergenceBounds.push_back({tZ, tR});
            fElCentralConvergenceBounds.push_front({tZ, -tR});

            tR = fRMax;
            for (const auto& sp : tElZHSolver->GetContainer()->RemoteSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                // if (dz < rho)
                //     continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h < tR)
                    tR = h;
            }

            vismsg_debug("  electric field remote radius at z=" << tZ << " is r" << (tR >= fRMax ? ">" : "=") << tR << " m" << eom);
            fElRemoteConvergenceBounds.push_back({tZ, tR});
            fElRemoteConvergenceBounds.push_front({tZ, -tR});
        }

        if (tMagZHSolver != nullptr) {
            double tR = 0;
            for (const auto& sp : tMagZHSolver->GetContainer()->CentralSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                if (dz > rho)
                    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h > tR)
                    tR = h;
            }

            vismsg_debug("  magnetic field central radius at z=" << tZ << " is r" << (tR >= fRMax ? ">" : "=") << tR << " m" << eom);
            fMagCentralConvergenceBounds.push_back({tZ, tR});
            fMagCentralConvergenceBounds.push_front({tZ, -tR});

            tR = fRMax;
            for (const auto& sp : tMagZHSolver->GetContainer()->RemoteSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                //if (dz < rho)
                //    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h < tR)
                    tR = h;
            }

            vismsg_debug("  magnetic field remote radius at z=" << tZ << " is r" << (tR >= fRMax ? ">" : "=") << tR << " m" << eom);
            fMagRemoteConvergenceBounds.push_back({tZ, tR});
            fMagRemoteConvergenceBounds.push_front({tZ, -tR});
        }
    }

    return;
}

void KSROOTZonalHarmonicsPainter::Display()
{
    if (! fDisplayEnabled)
        return;

    // the draw order is important!
    // correct order: canvas background -> remote radii -> central radii -> convergence area -> source points

    auto * tCanvas = GetWindow()->GetCanvas();;

    // draw background for remote source point radii
    if (fDrawRemoteBoundary) {
        tCanvas->SetFillStyle(kSolid);
        tCanvas->SetFillColorAlpha(kMagenta-3, 1.0);
    }

    // draw remote source point radii (inverse drawing on background)
    if (fDrawRemoteBoundary) {
        auto *tEllipse = new TEllipse();
        tEllipse->SetLineWidth(1);
        tEllipse->SetLineColorAlpha(kBlack, 0.05);
        tEllipse->SetFillStyle(kSolid);
        tEllipse->SetFillColorAlpha(kWhite, 0.1);

        for (auto & point : fElRemoteSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tEllipse->DrawEllipse(point.first, 0., point.second, 0., 0., 360., 0.);
        }

        for (auto & point : fMagRemoteSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tEllipse->DrawEllipse(point.first, 0., point.second, 0., 0., 360., 0.);
        }
    }

    // draw central source point radii
    if (fDrawCentralBoundary) {
        auto *tEllipse = new TEllipse();
        tEllipse->SetLineWidth(1);
        tEllipse->SetLineColorAlpha(kBlack, 0.005);
        tEllipse->SetFillStyle(kSolid);
        tEllipse->SetFillColorAlpha(kGreen-3, 0.01);

        for (auto & point : fElCentralSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tEllipse->DrawEllipse(point.first, 0., point.second, 0., 0., 360., 0.);
        }

        for (auto & point : fMagCentralSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tEllipse->DrawEllipse(point.first, 0., point.second, 0., 0., 360., 0.);
        }
    }

    if (fDrawConvergenceArea) {
        if (! fElCentralConvergenceBounds.empty()) {
            auto * tGraph = new TGraph();
            tGraph->SetLineWidth(3);
            tGraph->SetLineColor(kRed+1);

            for (auto & point : fElCentralConvergenceBounds) {
                tGraph->AddPoint(point.first, point.second);
            }
            tGraph->Draw("L");
        }

        if (! fMagCentralConvergenceBounds.empty()) {
            auto * tGraph = new TGraph();
            tGraph->SetLineWidth(3);
            tGraph->SetLineColor(kBlue+1);

            for (auto & point : fMagCentralConvergenceBounds) {
                tGraph->AddPoint(point.first, point.second);
            }
            tGraph->Draw("L");
        }

        if (! fMagRemoteConvergenceBounds.empty()) {
            auto * tGraph = new TGraph();
            tGraph->SetLineWidth(2);
            tGraph->SetLineColor(kBlue+1);
            tGraph->SetLineStyle(kDashed);

            for (auto & point : fMagRemoteConvergenceBounds) {
                tGraph->AddPoint(point.first, point.second);
            }
            tGraph->Draw("L");
        }
    }

    if (fDrawSourcePoints) {
        auto *tMarker = new TMarker();
        tMarker->SetMarkerColor(kBlack);
        tMarker->SetMarkerSize(.7);
        tMarker->SetMarkerStyle(kFullCircle);

        for (auto & point : fElCentralSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tMarker->DrawMarker(point.first, 0.);
        }

        for (auto & point : fMagCentralSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tMarker->DrawMarker(point.first, 0.);
        }

        tMarker->SetMarkerStyle(kOpenSquare);

        for (auto & point : fElRemoteSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tMarker->DrawMarker(point.first, 0.);
        }

        for (auto & point : fMagRemoteSourcePoints) {
            if ((point.first >= fZMin) && (point.first <= fZMax))
                tMarker->DrawMarker(point.first, 0.);
        }
    }

    return;
}

void KSROOTZonalHarmonicsPainter::Write()
{
    if (fWriteEnabled == false)
        return;

#if 0
        std::string tFile;

        if( fFile.length() > 0 )
        {
            if( fPath.empty() )
            {
                tFile = string( OUTPUT_DEFAULT_DIR ) + string("/") + fFile;
            }
            else
            {
                tFile = fPath + string("/") + fFile;
            }
        }
        else
        {
            tFile = string( OUTPUT_DEFAULT_DIR ) + string("/") + GetName() + string( ".xml" );
        }

        vismsg( eNormal ) << "writing convergence region to file <" << tFile << ">" << eom;
        std::ofstream tGeometryOutput( tFile.c_str() );
        vismsg( eNormal ) << "write mode is " << fGeometryType << eom;

        if( fGeometryType == "surface" )
        {
            tGeometryOutput << "<geometry>" << "\n";
            tGeometryOutput << "    <tag name=\"zh_electric_convergence\">" << "\n";
            tGeometryOutput << "        <rotated_poly_line_surface name=\"zh_electric_convergence_surface\">" << "\n";
            tGeometryOutput << "            <poly_line>" << "\n";
            tGeometryOutput << "                <start_point x=\"" << fZRPoints.at(0).first <<"\" y=\"" << fZRPoints.at(0).second << "\"/>" << "\n";

            for( unsigned int i=1; i<fZRPoints.size(); i++)
            {
                tGeometryOutput << "                <next_line x=\"" << fZRPoints.at(i).first <<"\" y=\"" << fZRPoints.at(i).second << "\"/>" << "\n";
            }

            tGeometryOutput << "            </poly_line>" << "\n";
            tGeometryOutput << "        </rotated_poly_line_surface>" << "\n";
            tGeometryOutput << "    </tag>" << "\n";
            tGeometryOutput << "</geometry>" << "\n";
            tGeometryOutput.close();

            return;

        } else if ( fGeometryType == "volume" ) {

            tGeometryOutput << "<geometry>" << "\n";
            tGeometryOutput << "    <tag name=\"zh_electric_convergence_volume\">" << "\n";

            for( unsigned int i=1; i<fZRPoints.size(); i++)
            {
                if ( fZRPoints.at(i-1).second != fZRPoints.at(i).second )
                {
                    tGeometryOutput << "     <cut_cone_space name=\"zh_generation_volume_" << i << "\"";
                    tGeometryOutput << " z1=\"" << fZRPoints.at(i-1).first;
                    tGeometryOutput << "\" z2=\"" << fZRPoints.at(i).first;
                    tGeometryOutput << "\" r1=\"" << fZRPoints.at(i-1).second;
                    tGeometryOutput << "\" r2=\"" << fZRPoints.at(i).second;
                    tGeometryOutput << "\"/>" << "\n";
                } else {
                    tGeometryOutput << "     <cylinder_space name=\"zh_generation_volume_" << i << "\"";
                    tGeometryOutput << " z1=\"" << fZRPoints.at(i-1).first;
                    tGeometryOutput << "\" z2=\"" << fZRPoints.at(i).first;
                    tGeometryOutput << "\" r=\"" << fZRPoints.at(i-1).second;
                    tGeometryOutput << "\"/>" << "\n";
                }

            }

            tGeometryOutput << "    </tag>" << "\n";
            tGeometryOutput << "<space name=\"zh_generation_assembly\">" << "\n";

            for( unsigned int i=1; i<fZRPoints.size(); i++)
            {
                tGeometryOutput << "<space name=\"placed_generation_volume_" << i << "\" node=\"zh_generation_volume_" << i << "\"/>\n";
            }

            tGeometryOutput << "</space>\n";
            tGeometryOutput << "</geometry>" << "\n";

            tGeometryOutput.close();
            return;

        } else {
            vismsg(eWarning) << "wrong fGeometryType for root_zh_painter, cannot write output!" << eom;
        }

        tGeometryOutput.close();
#endif

    return;
}

double KSROOTZonalHarmonicsPainter::GetXMin()
{
    return fZMin;
}
double KSROOTZonalHarmonicsPainter::GetXMax()
{
    return fZMax;
}

double KSROOTZonalHarmonicsPainter::GetYMin()
{
    return fRMin;
}
double KSROOTZonalHarmonicsPainter::GetYMax()
{
    return fRMax;
}

std::string KSROOTZonalHarmonicsPainter::GetXAxisLabel()
{
    return fXAxis;
}

std::string KSROOTZonalHarmonicsPainter::GetYAxisLabel()
{
    return fYAxis;
}

}  // namespace Kassiopeia
