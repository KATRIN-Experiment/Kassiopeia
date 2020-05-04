#include "KSROOTPotentialPainter.h"

#include "KSElectricField.h"
#include "KSFieldFinder.h"
#include "KSObject.h"
#include "KSVisualizationMessage.h"
#include "KToolbox.h"

#include <fstream>
#include <iostream>
#include <limits>

namespace Kassiopeia
{
KSROOTPotentialPainter::KSROOTPotentialPainter() : fXAxis("z"), fYAxis("y"), fCalcPot(true), fMap(), fComparison(false)
{}
KSROOTPotentialPainter::~KSROOTPotentialPainter() {}

void KSROOTPotentialPainter::Render()
{
    vismsg(eNormal) << "Getting electric field " << fElectricFieldName << " from the toolbox" << eom;
    KSElectricField* tElField = getElectricField(fElectricFieldName);
    if (tElField == nullptr)
        vismsg(eError) << "No electric Field!" << eom;
    vismsg(eNormal) << "Initialize electric field (again)" << eom;
    tElField->Initialize();

    // reference field
    KSElectricField* tRefField = nullptr;
    if (fComparison) {
        vismsg(eNormal) << "Getting reference electric field " << fReferenceFieldName << " from the toolbox" << eom;
        tRefField = getElectricField(fReferenceFieldName);

        if (tRefField == nullptr)
            vismsg(eError) << "No electric Field!" << eom;

        vismsg(eNormal) << "Initialize electric field (again)" << eom;
        tRefField->Initialize();
    }

    double tDeltaZ = fabs(fZmax - fZmin) / fZsteps;
    double tDeltaR = fabs(fRmax) / fRsteps;
    double tZ, tR;
    auto* Map = new TH2D("Map", "Map", fZsteps, fZmin, fZmax, 2 * fRsteps, -fRmax, fRmax);
    KThreeVector tPosition;

    KThreeVector ElectricField;
    Double_t tPotential;

    KThreeVector tRefElectricField;
    Double_t tRefPotential;

    Double_t tRelError = 0.;

    vismsg(eNormal) << "start calculating potential map" << eom;
    for (int i = 0; i < fZsteps; i++) {
        tZ = fZmin + i * tDeltaZ;
        vismsg(eNormal) << "Electric Field: Z Position: " << i << "/" << fZsteps << reom;

        for (int j = fRsteps; j >= 0; j--) {
            tR = j * tDeltaR;

            if (fCalcPot == 0) {
                if (!fComparison) {
                    if (fYAxis == "y")
                        tPosition.SetComponents(0., -tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(-tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculateField(tPosition, 0.0, ElectricField);
                    Map->SetBinContent(i + 1, fRsteps - j + 1, ElectricField.Magnitude());

                    if (fYAxis == "y")
                        tPosition.SetComponents(0., tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculateField(tPosition, 0.0, ElectricField);
                    Map->SetBinContent(i + 1, fRsteps + j + 1, ElectricField.Magnitude());
                }
                else {
                    if (fYAxis == "y")
                        tPosition.SetComponents(0., -tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(-tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculateField(tPosition, 0.0, ElectricField);
                    tRefField->CalculateField(tPosition, 0.0, tRefElectricField);

                    tRelError =
                        fabs((ElectricField.X() - tRefElectricField.X()) + (ElectricField.Y() - tRefElectricField.Y()) +
                             (ElectricField.Z() - tRefElectricField.Z()));
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tRelError);

                    if (fYAxis == "y")
                        tPosition.SetComponents(0., tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculateField(tPosition, 0.0, ElectricField);
                    tRefField->CalculateField(tPosition, 0.0, tRefElectricField);

                    tRelError =
                        fabs((ElectricField.X() - tRefElectricField.X()) + (ElectricField.Y() - tRefElectricField.Y()) +
                             (ElectricField.Z() - tRefElectricField.Z()));
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tRelError);
                }
            }
            else {
                if (!fComparison) {
                    if (fYAxis == "y")
                        tPosition.SetComponents(0., -tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(-tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculatePotential(tPosition, 0.0, tPotential);
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tPotential);

                    if (fYAxis == "y")
                        tPosition.SetComponents(0., tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculatePotential(tPosition, 0.0, tPotential);
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tPotential);
                }
                else {
                    if (fYAxis == "y")
                        tPosition.SetComponents(0., -tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(-tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculatePotential(tPosition, 0.0, tPotential);
                    tRefField->CalculatePotential(tPosition, 0.0, tRefPotential);
                    tRelError = fabs(tPotential - tRefPotential);
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tRelError);

                    if (fYAxis == "y")
                        tPosition.SetComponents(0., tR, tZ);
                    else if (fYAxis == "x")
                        tPosition.SetComponents(tR, 0., tZ);
                    else
                        vismsg(eError)
                            << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                            << eom;

                    tElField->CalculatePotential(tPosition, 0.0, tPotential);
                    tRefField->CalculatePotential(tPosition, 0.0, tRefPotential);
                    tRelError = fabs(tPotential - tRefPotential);
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tRelError);
                }
            }
        }
    }

    fMap = Map;

    return;
}

void KSROOTPotentialPainter::Display()
{
    if (fDisplayEnabled == true) {
        fWindow->GetPad()->SetRightMargin(0.15);
        if (fCalcPot == 1)
            fMap->SetZTitle("Potential (V)");
        else
            fMap->SetZTitle("Electric Field (V/m)");
        if (fXAxis == "z")
            fMap->GetXaxis()->SetTitle("z (m)");
        else if (fXAxis == "y")
            fMap->GetXaxis()->SetTitle("y (m)");
        else if (fXAxis == "x")
            fMap->GetXaxis()->SetTitle("x (m)");
        if (fYAxis == "z")
            fMap->GetYaxis()->SetTitle("z (m)");
        else if (fYAxis == "y")
            fMap->GetYaxis()->SetTitle("y (m)");
        else if (fYAxis == "x")
            fMap->GetYaxis()->SetTitle("x (m)");

        fMap->GetZaxis()->SetTitleOffset(1.4);
        fMap->SetStats(false);
        fMap->SetTitle("");
        fMap->Draw("COLZL");
    }
    return;
}

void KSROOTPotentialPainter::Write() {}

double KSROOTPotentialPainter::GetXMin()
{
    double tMin(std::numeric_limits<double>::max());
    return tMin;
}
double KSROOTPotentialPainter::GetXMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    return tMax;
}

double KSROOTPotentialPainter::GetYMin()
{
    double tMin(std::numeric_limits<double>::max());
    return tMin;
}
double KSROOTPotentialPainter::GetYMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    return tMax;
}

std::string KSROOTPotentialPainter::GetXAxisLabel()
{
    return fXAxis;
}

std::string KSROOTPotentialPainter::GetYAxisLabel()
{
    return fYAxis;
}

}  // namespace Kassiopeia
