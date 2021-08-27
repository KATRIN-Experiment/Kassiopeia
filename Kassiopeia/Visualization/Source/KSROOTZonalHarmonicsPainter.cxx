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
    fZmin(0.0),
    fZmax(0.0),
    fRmin(0.0),
    fRmax(5.0),
    fZdist(0.01),
    fRdist(0.001),
    fZMaxSteps(1000),
    fRMaxSteps(1000),
    fElectricFieldName(""),
    fMagneticFieldName(""),
    fFile(""),
    fPath(""),
    fDrawSourcePoints(true),
    fDrawExpansionArea(false),
    fDrawConvergenceArea(true)
{}
KSROOTZonalHarmonicsPainter::~KSROOTZonalHarmonicsPainter() = default;

void KSROOTZonalHarmonicsPainter::Render()
{
    bool autoAdjustZ = (fZmin >= fZmax);
    if (autoAdjustZ) {
        fZmin = std::numeric_limits<double>::max();
        fZmax = std::numeric_limits<double>::min();
    }

    fElZHPoints = new TPolyMarker();
    fMagZHPoints = new TPolyMarker();

    //fZRPoints.clear();

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
        if (! tMagField) {
            KSMagneticField* tFieldWrapper = getMagneticField(fMagneticFieldName);
            auto* tKEMFieldObject = dynamic_cast<KSMagneticKEMField*>(tFieldWrapper);
            tMagField = dynamic_cast<KGStaticElectromagnetField*>(tKEMFieldObject->GetMagneticField());
        }

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

    unsigned tNPoints = floor((fZmax - fZmin) / fZdist);

    fElZHConvergenceGraph = new TGraph(2*tNPoints);
    fElZHCentralGraph = new TGraph(2*tNPoints);
    fElZHRemoteGraph = new TGraph(2*tNPoints);

    fMagZHConvergenceGraph = new TGraph(2*tNPoints);
    fMagZHCentralGraph = new TGraph(2*tNPoints);
    fMagZHRemoteGraph = new TGraph(2*tNPoints);

    vismsg(eNormal) << "ZH painter: start calculating convergence boundary from " << fZmin << " to " << fZmax << " m ("
                    << tNPoints << " steps) ..." << eom;

    for (unsigned tPointIndex = 0; tPointIndex < tNPoints; tPointIndex++) {
        unsigned tNegPointIndex = 2*tNPoints - tPointIndex - 1;
        double tZ = fZmin + tPointIndex * fZdist;

        if (tElZHSolver != nullptr) {
            double tR = 0;
            for (auto& sp : tElZHSolver->CentralSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                if (dz > rho)
                    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h > tR)
                    tR = h;
            }

            //vismsg(eDebug) << "  electric field sourcepoint radius at z=" << tZ << " is r=" << tR << " m" << eom;
            fElZHConvergenceGraph->SetPoint(tPointIndex, tZ, tR);
            fElZHConvergenceGraph->SetPoint(tNegPointIndex, tZ, -tR);

            fElZHCentralGraph->SetPoint(tPointIndex, tZ, 0);
            fElZHCentralGraph->SetPoint(tNegPointIndex, tZ, 0);
            fElZHRemoteGraph->SetPoint(tPointIndex, tZ, 0);
            fElZHRemoteGraph->SetPoint(tNegPointIndex, tZ, 0);

            // scan central convergence region
            for (double tRho = fRmin; tRho <= tR; tRho += fRdist) {
                if (!tElZHSolver->UseCentralExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  electric field central convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    break;
                }
                fElZHCentralGraph->SetPoint(tPointIndex, tZ, tRho);
                fElZHCentralGraph->SetPoint(tNegPointIndex, tZ, -tRho);
            }
            // scan remote convergence region
            for (double tRho = fRmax; tRho >= tR; tRho -= fRdist) {
                if (!tElZHSolver->UseRemoteExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  electric field remote convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    break;
                }
                fElZHRemoteGraph->SetPoint(tPointIndex, tZ, tRho);
                fElZHRemoteGraph->SetPoint(tNegPointIndex, tZ, -tRho);
            }
        }

        if (tMagZHSolver != nullptr) {
            double tR = 0;
            for (auto& sp : tMagZHSolver->CentralSourcePoints()) {
                double rho = sp.second;
                double dz = fabs(tZ - sp.first);
                if (dz > rho)
                    continue;

                double h = sqrt(rho * rho - dz * dz);
                if (h > tR)
                    tR = h;
            }

            //vismsg(eDebug) << "  magnetic field sourcepoint radius at z=" << tZ << " is r=" << tR << " m" << eom;
            fMagZHConvergenceGraph->SetPoint(tPointIndex, tZ, tR);
            fMagZHConvergenceGraph->SetPoint(tNegPointIndex, tZ, -tR);

            fMagZHCentralGraph->SetPoint(tPointIndex, tZ, 0);
            fMagZHCentralGraph->SetPoint(tNegPointIndex, tZ, 0);
            fMagZHRemoteGraph->SetPoint(tPointIndex, tZ, 0);
            fMagZHRemoteGraph->SetPoint(tNegPointIndex, tZ, 0);

            // scan central convergence region
            for (double tRho = fRmin; tRho <= tR; tRho += fRdist) {
                if (!tMagZHSolver->UseCentralExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  magnetic field central convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    break;
                }
                fMagZHCentralGraph->SetPoint(tPointIndex, tZ, tRho);
                fMagZHCentralGraph->SetPoint(tNegPointIndex, tZ, -tRho);
            }
            // scan central remote region
            for (double tRho = fRmax; tRho >= tR; tRho -= fRdist) {
                if (!tMagZHSolver->UseRemoteExpansion(KThreeVector(0, tRho, tZ))) {
                    vismsg(eDebug) << "  magnetic field remote convergence limit at z=" << tZ << " is r=" << tRho
                                   << " m" << eom;
                    break;
                }
                fMagZHRemoteGraph->SetPoint(tPointIndex, tZ, tRho);
                fMagZHRemoteGraph->SetPoint(tNegPointIndex, tZ, -tRho);
            }
        }
    }

    if (tElZHSolver != nullptr) {
        fElZHConvergenceGraph->SetLineColor(kRed);

        fElZHCentralGraph->SetLineColor(kRed);
        fElZHCentralGraph->SetLineStyle(kDotted);
        fElZHRemoteGraph->SetLineColor(kRed);
        fElZHRemoteGraph->SetLineStyle(kDashed);

        fElZHPoints->SetMarkerColor(kBlack);
        fElZHPoints->SetMarkerStyle(kFullDotSmall);
    }

    if (tMagZHSolver != nullptr) {
        fMagZHConvergenceGraph->SetLineColor(kBlue);

        fMagZHCentralGraph->SetLineColor(kBlue);
        fMagZHCentralGraph->SetLineStyle(kDotted);
        fMagZHRemoteGraph->SetLineColor(kBlue);
        fMagZHRemoteGraph->SetLineStyle(kDashed);

        fMagZHPoints->SetMarkerColor(kBlack);
        fMagZHPoints->SetMarkerStyle(kFullDotSmall);
    }

    return;
}

void KSROOTZonalHarmonicsPainter::Display()
{
    if (fDisplayEnabled == true) {
        // draw order may be important, who knows what ROOT does anyway ...

        auto tGraphs = new TMultiGraph();

        // TODO: make this an option
        if (fDrawExpansionArea) {
            if (fElZHCentralGraph->GetN() > 0)
                tGraphs->Add(fElZHCentralGraph);
            if (fMagZHCentralGraph->GetN() > 0)
                tGraphs->Add(fMagZHCentralGraph);

            // FIXME: this is confusing
            if (fElZHRemoteGraph->GetN() > 0)
               tGraphs->Add(fElZHRemoteGraph);
            if (fMagZHRemoteGraph->GetN() > 0)
               tGraphs->Add(fMagZHRemoteGraph);
        }

        // TODO: make this an option
        if (fDrawConvergenceArea) {
            if (fElZHConvergenceGraph->GetN() > 0)
                tGraphs->Add(fElZHConvergenceGraph);
            if (fMagZHConvergenceGraph->GetN() > 0)
                tGraphs->Add(fMagZHConvergenceGraph);
        }

        tGraphs->Draw("L");

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
