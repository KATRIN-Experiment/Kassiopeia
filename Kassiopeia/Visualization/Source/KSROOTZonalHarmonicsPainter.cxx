#include "KSROOTZonalHarmonicsPainter.h"

#include "KElectricZHFieldSolver.hh"
#include "KGElectrostaticBoundaryField.hh"
#include "KGStaticElectromagnetField.hh"
#include "KSObject.h"
#include "KSReadFileROOT.h"
#include "KSVisualizationMessage.h"
#include "KToolbox.h"
#include "KZonalHarmonicMagnetostaticFieldSolver.hh"
#include "TMultiGraph.h"

#include <fstream>
#include <iostream>
#include <limits>

using namespace KEMField;

namespace
{

void MirrorGraph(TGraph* g)
{
    // used to shade area between graphs, see: https://root.cern.ch/root/html/tutorials/graphics/graphShade.C.html
    if (g->GetN() <= 0)
        return;

    int n = g->GetN();
    double* x = g->GetX();
    double* y = g->GetY();
    for (int i = 0; i < n; i++) {
        g->SetPoint(2 * n - i, x[i], -y[i]);
    }
}

}  // namespace

namespace Kassiopeia
{
KSROOTZonalHarmonicsPainter::KSROOTZonalHarmonicsPainter() :
    fXAxis("z"),
    fYAxis("r"),
    fZmin(0.0),
    fZmax(0.0),
    fRmin(0.0),
    fRmax(5.0),
    fZdist(0.01),
    fRdist(0.001),
    fZMaxSteps(5000),
    fRMaxSteps(5000),
    fElectricFieldName(""),
    fMagneticFieldName(""),
    fFile(""),
    fPath(""),
    fDrawSourcePoints(true),
    fDrawSourcePointArea(true)
//fGeometryType ( "surface" ),
//fRadialSafetyMargin( 0. ),
//fZRPoints ()
{}
KSROOTZonalHarmonicsPainter::~KSROOTZonalHarmonicsPainter() {}

void KSROOTZonalHarmonicsPainter::Render()
{
    bool autoAdjustZ = (fZmin >= fZmax);
    if (autoAdjustZ) {
        fZmin = std::numeric_limits<double>::max();
        fZmax = std::numeric_limits<double>::min();
    }

    fElZHConvergenceGraph = new TGraph();
    fElZHCentralGraph = new TGraph();
    fElZHRemoteGraph = new TGraph();

    fMagZHConvergenceGraph = new TGraph();
    fMagZHCentralGraph = new TGraph();
    fMagZHRemoteGraph = new TGraph();

    fElZHPoints = new TPolyMarker();
    fMagZHPoints = new TPolyMarker();

    //fZRPoints.clear();

    KElectricZHFieldSolver* tElZHSolver = nullptr;
    if (!fElectricFieldName.empty()) {
        vismsg(eNormal) << "Getting electric field " << fElectricFieldName << " from the toolbox" << eom;
        auto* tElField = katrin::KToolbox::GetInstance().Get<KGElectrostaticBoundaryField>(fElectricFieldName);
        if (tElField == nullptr)
            vismsg(eError) << "No electric Field!" << eom;
        vismsg(eNormal) << "Initialize electric field (again)" << eom;
        tElField->Initialize();

        //vismsg(eNormal) << "retrieve converter from electric field" << eom;
        auto tElConverter = tElField->GetConverter();
        if (tElConverter.Null())
            vismsg(eError) << "Electric Field has no Converter! " << eom;

        //vismsg(eNormal) << "retrieve ZH solver from electric field" << eom;
        tElZHSolver = dynamic_cast<KElectricZHFieldSolver*>(&(*tElField->GetFieldSolver()));
        if (tElZHSolver == nullptr)
            vismsg(eError) << "Electric Field has no ZHSolver!" << eom;

        if (autoAdjustZ) {
            auto z1 = tElZHSolver->GetParameters()->GetCentralZ1();
            auto z2 = tElZHSolver->GetParameters()->GetCentralZ2();
            if (z1 < fZmin)
                fZmin = z1;
            if (z2 > fZmax)
                fZmax = z2;
        }

        for (auto& sp : tElZHSolver->CentralSourcePoints()) {
            fElZHPoints->SetNextPoint(sp.first, 0);
        }
    }

    KZonalHarmonicMagnetostaticFieldSolver* tMagZHSolver = nullptr;
    if (!fMagneticFieldName.empty()) {
        vismsg(eNormal) << "Getting magnetic field " << fMagneticFieldName << " from the toolbox" << eom;
        auto* tMagField = katrin::KToolbox::GetInstance().Get<KGStaticElectromagnetField>(fMagneticFieldName);
        if (tMagField == nullptr)
            vismsg(eError) << "No magnetic Field!" << eom;
        vismsg(eNormal) << "Initialize magnetic field (again)" << eom;
        tMagField->Initialize();

        //vismsg(eNormal) << "retrieve converter from magnetic field" << eom;
        //auto tMagConverter = tMagField->GetConverter();
        //if ( tMagConverter.Null() )
        //    vismsg(eError) << "Magnetic Field has no Converter! " << eom;

        //vismsg(eNormal) << "retrieve ZH solver from magnetic field" << eom;
        tMagZHSolver = dynamic_cast<KZonalHarmonicMagnetostaticFieldSolver*>(&(*tMagField->GetFieldSolver()));
        if (tMagZHSolver == nullptr)
            vismsg(eError) << "Magnetic Field has no ZHSolver!" << eom;

        if (autoAdjustZ) {
            auto z1 = tMagZHSolver->GetParameters()->GetCentralZ1();
            auto z2 = tMagZHSolver->GetParameters()->GetCentralZ2();
            if (z1 < fZmin)
                fZmin = z1;
            if (z2 > fZmax)
                fZmax = z2;
        }

        for (auto& sp : tMagZHSolver->CentralSourcePoints()) {
            fMagZHPoints->SetNextPoint(sp.first, 0);
        }
    }

    double tDeltaZ = fZdist;
    if (ceil(fabs(fZmax - fZmin) / tDeltaZ) > fZMaxSteps) {
        tDeltaZ = fabs(fZmax - fZmin) / fZMaxSteps;
    }
    double tDeltaR = fRdist;
    if (ceil(fabs(fRmax - fRmin) / tDeltaR) > fRMaxSteps) {
        tDeltaR = fabs(fRmax - fRmin) / fRMaxSteps;
    }

    KThreeVector tPosition;

    vismsg(eNormal) << "ZH painter: start calculating convergence boundary from " << fZmin << " to " << fZmax << " m"
                    << eom;
    for (double tZ = fZmin; tZ <= fZmax; tZ += fZdist) {
        //vismsg( eInfo ) << "ZH painter: Z Position: " << tZ << eom;
        double tR = 0;

        if (tElZHSolver != nullptr) {
            for (auto& sp : tElZHSolver->CentralSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                if (dz > rho)
                    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h > tR)
                    tR = h;
            }

            if (tR >= 0) {
                //vismsg(eDebug) << "  electric field sourcepoint radius at z=" << tZ << " is r=" << tR << " m" << eom;
                fElZHConvergenceGraph->SetPoint(fElZHConvergenceGraph->GetN(), tZ, tR);
            }

            // scan central convergence region
            for (double tRho = 0; tRho <= tR; tRho += fRdist) {
                if (!tElZHSolver->UseCentralExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  electric field central convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    fElZHCentralGraph->SetPoint(fElZHCentralGraph->GetN(), tZ, tRho - fRdist);
                    break;
                }
            }
            // scan remote convergence region
            for (double tRho = tR; tRho >= 0; tRho -= fRdist) {
                if (!tElZHSolver->UseRemoteExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  electric field remote convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    fElZHRemoteGraph->SetPoint(fElZHRemoteGraph->GetN(), tZ, tRho + fRdist);
                    break;
                }
            }
        }

        if (tMagZHSolver != nullptr) {
            for (auto& sp : tMagZHSolver->CentralSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                if (dz > rho)
                    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h > tR)
                    tR = h;
            }

            if (tR >= 0) {
                //vismsg(eDebug) << "  magnetic field sourcepoint radius at z=" << tZ << " is r=" << tR << " m" << eom;
                fMagZHConvergenceGraph->SetPoint(fMagZHConvergenceGraph->GetN(), tZ, tR);
            }

            // scan central convergence region
            for (double tRho = 0; tRho <= tR; tRho += fRdist) {
                if (!tMagZHSolver->UseCentralExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  magnetic field central convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    fMagZHCentralGraph->SetPoint(fMagZHCentralGraph->GetN(), tZ, tRho - fRdist);
                    break;
                }
            }
            // scan central remote region
            for (double tRho = tR; tRho >= 0; tRho -= fRdist) {
                if (!tMagZHSolver->UseRemoteExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  magnetic field remote convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    fMagZHRemoteGraph->SetPoint(fMagZHRemoteGraph->GetN(), tZ, tRho + fRdist);
                    break;
                }
            }
        }
    }

    if (tElZHSolver != nullptr) {
        // add lower part of graphs (negative R) for filled draw, points in reverse order
        MirrorGraph(fElZHConvergenceGraph);
        MirrorGraph(fElZHCentralGraph);
        MirrorGraph(fElZHRemoteGraph);

        fElZHConvergenceGraph->SetLineColor(kRed);
        fElZHConvergenceGraph->SetLineStyle(kDotted);

        fElZHCentralGraph->SetLineColor(kRed);
        fElZHCentralGraph->SetFillColorAlpha(kRed, 0.1);

        fElZHRemoteGraph->SetLineColor(kRed);

        fElZHPoints->SetMarkerColor(kRed);
        fElZHPoints->SetMarkerStyle(kFullDotMedium);
    }

    if (tMagZHSolver != nullptr) {
        // add lower part of graph (negative R) for filled draw, points in reverse order
        MirrorGraph(fMagZHConvergenceGraph);
        MirrorGraph(fMagZHCentralGraph);
        MirrorGraph(fMagZHRemoteGraph);

        fMagZHConvergenceGraph->SetLineColor(kBlue);
        fMagZHConvergenceGraph->SetLineStyle(kDotted);

        fMagZHCentralGraph->SetLineColor(kBlue);
        fMagZHCentralGraph->SetFillColorAlpha(kBlue, 0.1);

        fMagZHRemoteGraph->SetLineColor(kBlue);

        fMagZHPoints->SetMarkerColor(kBlue);
        fMagZHPoints->SetMarkerStyle(kFullDotMedium);
    }

    return;
}

void KSROOTZonalHarmonicsPainter::Display()
{
    if (fDisplayEnabled == true) {
        // draw order may be important, who knows what ROOT does anyway ...

        auto tGraphsFilled = new TMultiGraph();
        auto tGraphs = new TMultiGraph();

        if (fElZHCentralGraph->GetN() > 0)
            tGraphsFilled->Add(fElZHCentralGraph);
        if (fMagZHCentralGraph->GetN() > 0)
            tGraphsFilled->Add(fMagZHCentralGraph);

        // FIXME: this is confusing
        //if (fElZHRemoteGraph->GetN() > 0)
        //    tGraphs->Add(fElZHRemoteGraph);
        //if (fMagZHRemoteGraph->GetN() > 0)
        //    tGraphs->Add(fMagZHRemoteGraph);

        // TODO: make this an option
        if (fDrawSourcePointArea) {
            if (fElZHConvergenceGraph->GetN() > 0)
                tGraphs->Add(fElZHConvergenceGraph);
            if (fMagZHConvergenceGraph->GetN() > 0)
                tGraphs->Add(fMagZHConvergenceGraph);
        }

        tGraphsFilled->Draw("fl");
        tGraphs->Draw("l");

        // TODO: make this an option
        if (fDrawSourcePoints) {
            if (fElZHPoints->GetN() > 0)
                fElZHPoints->Draw();
            if (fMagZHPoints->GetN() > 0)
                fMagZHPoints->Draw();
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
    double tMin(std::numeric_limits<double>::max());
    return tMin;
}
double KSROOTZonalHarmonicsPainter::GetXMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    return tMax;
}

double KSROOTZonalHarmonicsPainter::GetYMin()
{
    double tMin(std::numeric_limits<double>::max());
    return tMin;
}
double KSROOTZonalHarmonicsPainter::GetYMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    return tMax;
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
