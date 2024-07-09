#include "KSROOTMagFieldPainter.h"

#include "KSFieldFinder.h"
#include "KSObject.h"
#include "KSVisualizationMessage.h"
#include "KToolbox.h"
#include "TColor.h"
#include "TStyle.h"

#include <fstream>
#include <iostream>
#include <limits>

using namespace katrin;
using namespace std;

namespace Kassiopeia
{
KSROOTMagFieldPainter::KSROOTMagFieldPainter() :
    fXAxis("z"),
    fYAxis("y"),
    fZfix(0),
    fAxialSymmetry(true),
    fPlot("magnetic_field_abs"),
    fUseLogZ(false),
    fGradNumerical(true),
    fDraw("COLZ"),
    fMap()
{}
KSROOTMagFieldPainter::~KSROOTMagFieldPainter() = default;

void KSROOTMagFieldPainter::FieldMapZ(KSMagneticField* tMagField, double tDeltaZ, double tDeltaR)
{
    double tZ, tR;
    auto* Map = new TH2D("Map", "Map", fZsteps, fZmin, fZmax, 2 * fRsteps, -fRmax, fRmax);
    KThreeVector tPosition;
    KThreeVector tPosition_i, tPosition_j;

    KThreeVector tMagneticField;
    KThreeVector tPotential;
    KThreeMatrix tGradient;

    if (fAxialSymmetry == true) {
        vismsg(eNormal) << "start calculating <" << fPlot << "> map, (assuming axial symmetry!!)" << eom;
        for (int i = 0; i < fZsteps; i++) {
            tZ = fZmin + i * tDeltaZ;
            vismsg(eNormal) << "map: Z Position: " << i << "/" << fZsteps << reom;

            for (int j = fRsteps; j >= 0; j--) {
                tR = j * tDeltaR;

                if (fYAxis == "y") {
                    tPosition.SetComponents(0., tR, tZ);
                    tPosition_i.SetComponents(0., tR, tZ + tDeltaZ);
                    tPosition_j.SetComponents(0., tR + tDeltaR, tZ);
                }
                else if (fYAxis == "x") {
                    tPosition.SetComponents(tR, 0., tZ);
                    tPosition_i.SetComponents(tR, 0., tZ + tDeltaZ);
                    tPosition_j.SetComponents(tR + tDeltaR, 0., tZ);
                }
                else
                    vismsg(eError)
                        << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                        << eom;
                //					calculate magnetic field at tPosition, requested by any further calculation
                tMagField->CalculateField(tPosition, 0.0, tMagneticField);
                if (fPlot == "magnetic_field_abs") {
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tMagneticField.Magnitude());
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.Magnitude());
                }
                else if (fPlot == "magnetic_field_z") {
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tMagneticField.Z());
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.Z());
                }
                else if (fPlot == "magnetic_field_z_abs") {
                    Map->SetBinContent(i + 1, fRsteps - j + 1, fabs(tMagneticField.Z()));
                    Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.Z()));
                }
                else if (fPlot == "magnetic_field_x" || fPlot == "magnetic_field_y") {
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tMagneticField.X());
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.X());
                }
                else if (fPlot == "magnetic_field_x_abs" || fPlot == "magnetic_field_y_abs") {
                    Map->SetBinContent(i + 1, fRsteps - j + 1, fabs(tMagneticField.X()));
                    Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.X()));
                }
                else if (fPlot == "magnetic_potential_abs") {
                    tMagField->CalculatePotential(tPosition, 0.0, tPotential);
                    Map->SetBinContent(i + 1, fRsteps - j + 1, tPotential.Magnitude());
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tPotential.Magnitude());
                }
                else if (fPlot.compare(9, 8, "gradient") == 0) {
                    if (fGradNumerical == true) {
                        KThreeVector tMagneticField_i, tMagneticField_j;
                        if (fPlot == "magnetic_gradient_z") {
                            tMagField->CalculateField(tPosition_i, 0.0, tMagneticField_i);
                            Double_t tGradient = (tMagneticField_i.Magnitude() - tMagneticField.Magnitude()) / tDeltaZ;
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradient);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_z_abs") {
                            tMagField->CalculateField(tPosition_i, 0.0, tMagneticField_i);
                            Double_t tGradient =
                                fabs((tMagneticField_i.Magnitude() - tMagneticField.Magnitude()) / tDeltaZ);
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradient);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_x" || fPlot == "magnetic_gradient_y") {
                            tMagField->CalculateField(tPosition_j, 0.0, tMagneticField_j);
                            Double_t tGradient = (tMagneticField_j.Magnitude() - tMagneticField.Magnitude()) / tDeltaR;
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradient);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_x_abs" || fPlot == "magnetic_gradient_y_abs") {
                            tMagField->CalculateField(tPosition_j, 0.0, tMagneticField_j);
                            Double_t tGradient =
                                fabs((tMagneticField_j.Magnitude() - tMagneticField.Magnitude()) / tDeltaR);
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradient);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else {
                            vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                                           << "> is not defined in KSROOTMagFieldPainter" << eom;
                        }
                    }
                    else if (fGradNumerical == false) {
                        //							calculate gradient matrix
                        tMagField->CalculateGradient(tPosition, 0.0, tGradient);
                        KThreeVector tGradB;
                        tGradB.SetX(tMagneticField.X() * tGradient[0] + tMagneticField.Y() * tGradient[1] +
                                    tMagneticField.Z() * tGradient[2]);
                        tGradB.SetY(tMagneticField.X() * tGradient[3] + tMagneticField.Y() * tGradient[4] +
                                    tMagneticField.Z() * tGradient[5]);
                        tGradB.SetZ(tMagneticField.X() * tGradient[6] + tMagneticField.Y() * tGradient[7] +
                                    tMagneticField.Z() * tGradient[8]);
                        if (tMagneticField.Magnitude() > 0)
                            tGradB /= tMagneticField.Magnitude();
                        if (fPlot == "magnetic_gradient_abs") {
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradB.Magnitude());
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Magnitude());
                        }
                        else if (fPlot == "magnetic_gradient_z") {
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradB.Z());
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Z());
                        }
                        else if (fPlot == "magnetic_gradient_z_abs") {
                            Map->SetBinContent(i + 1, fRsteps - j + 1, fabs(tGradB.Z()));
                            Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.Z()));
                        }
                        else if (fPlot == "magnetic_gradient_x" || fPlot == "magnetic_gradient_y") {
                            Map->SetBinContent(i + 1, fRsteps - j + 1, tGradB.X());
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.X());
                        }
                        else if (fPlot == "magnetic_gradient_x_abs" || fPlot == "magnetic_gradient_y_abs") {
                            Map->SetBinContent(i + 1, fRsteps - j + 1, fabs(tGradB.X()));
                            Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.X()));
                        }
                        else {
                            vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                                           << "> is not defined in KSROOTMagFieldPainter" << eom;
                        }
                    }
                }
                else {
                    vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                                   << "> is not defined in KSROOTMagFieldPainter" << eom;
                }
            }
        }
    }
    else if (fAxialSymmetry == false) {
        vismsg(eNormal) << "start calculating <" << fPlot << "> map" << eom;
        KThreeVector tPosition_x, tPosition_y;
        for (int i = 0; i < fZsteps; i++) {
            tZ = fZmin + i * tDeltaZ;
            vismsg(eNormal) << "map: Z Position: " << i << "/" << fZsteps << reom;

            for (int j = fRsteps; j >= -fRsteps; j--) {
                tR = j * tDeltaR;

                tPosition_x.SetComponents(tR + tDeltaR, 0., tZ);
                tPosition_y.SetComponents(0., tR + tDeltaR, tZ);
                if (fYAxis == "y") {
                    tPosition.SetComponents(0., tR, tZ);
                    tPosition_i.SetComponents(0., tR, tZ + tDeltaZ);
                    tPosition_j.SetComponents(0., tR + tDeltaR, tZ);
                }
                else if (fYAxis == "x") {
                    tPosition.SetComponents(tR, 0., tZ);
                    tPosition_i.SetComponents(tR, 0., tZ + tDeltaZ);
                    tPosition_j.SetComponents(tR + tDeltaR, 0., tZ);
                }
                else
                    vismsg(eError)
                        << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included"
                        << eom;
                //					calculate magnetic field at tPosition, requested by any further calculation
                tMagField->CalculateField(tPosition, 0.0, tMagneticField);
                if (fPlot == "magnetic_field_abs") {
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.Magnitude());
                }
                else if (fPlot == "magnetic_field_z") {
                    tMagField->CalculateField(tPosition, 0.0, tMagneticField);
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.Z());
                }
                else if (fPlot == "magnetic_field_z_abs") {
                    Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.Z()));
                }
                else if (fPlot == "magnetic_field_x") {
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.X());
                }
                else if (fPlot == "magnetic_field_x_abs") {
                    Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.X()));
                }
                else if (fPlot == "magnetic_field_y") {
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.Y());
                }
                else if (fPlot == "magnetic_field_y_abs") {
                    Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.Y()));
                }
                else if (fPlot == "magnetic_potential_abs") {
                    tMagField->CalculatePotential(tPosition, 0.0, tPotential);
                    Map->SetBinContent(i + 1, fRsteps + j + 1, tPotential.Magnitude());
                }
                else if (fPlot.compare(9, 8, "gradient") == 0) {
                    KThreeVector tMagneticField_i, tMagneticField_j;
                    if (fGradNumerical == true) {
                        KThreeVector tMagneticField_x, tMagneticField_y;
                        if (fPlot == "magnetic_gradient_z") {
                            tMagField->CalculateField(tPosition_i, 0.0, tMagneticField_i);
                            Double_t tGradient = (tMagneticField_i.Magnitude() - tMagneticField.Magnitude()) / tDeltaZ;
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_z_abs") {
                            tMagField->CalculateField(tPosition_i, 0.0, tMagneticField_i);
                            Double_t tGradient =
                                fabs((tMagneticField_i.Magnitude() - tMagneticField.Magnitude()) / tDeltaZ);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_x") {
                            tMagField->CalculateField(tPosition_x, 0.0, tMagneticField_x);
                            Double_t tGradient = (tMagneticField_x.Magnitude() - tMagneticField.Magnitude()) / tDeltaR;
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_x_abs") {
                            tMagField->CalculateField(tPosition_j, 0.0, tMagneticField_x);
                            Double_t tGradient =
                                fabs((tMagneticField_x.Magnitude() - tMagneticField.Magnitude()) / tDeltaR);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_y") {
                            tMagField->CalculateField(tPosition_y, 0.0, tMagneticField_y);
                            Double_t tGradient = (tMagneticField_y.Magnitude() - tMagneticField.Magnitude()) / tDeltaR;
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else if (fPlot == "magnetic_gradient_y_abs") {
                            tMagField->CalculateField(tPosition_j, 0.0, tMagneticField_y);
                            Double_t tGradient =
                                fabs((tMagneticField_y.Magnitude() - tMagneticField.Magnitude()) / tDeltaR);
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                        }
                        else {
                            vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                                           << "> is not defined in KSROOTMagFieldPainter" << eom;
                        }
                    }
                    else if (fGradNumerical == false) {
                        //							calculate gradient matrix
                        tMagField->CalculateGradient(tPosition, 0.0, tGradient);
                        KThreeVector tGradB;
                        tGradB.SetX(tMagneticField.X() * tGradient[0] + tMagneticField.Y() * tGradient[1] +
                                    tMagneticField.Z() * tGradient[2]);
                        tGradB.SetY(tMagneticField.X() * tGradient[3] + tMagneticField.Y() * tGradient[4] +
                                    tMagneticField.Z() * tGradient[5]);
                        tGradB.SetZ(tMagneticField.X() * tGradient[6] + tMagneticField.Y() * tGradient[7] +
                                    tMagneticField.Z() * tGradient[8]);
                        if (tMagneticField.Magnitude() > 0)
                            tGradB /= tMagneticField.Magnitude();
                        if (fPlot == "magnetic_gradient_abs") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Magnitude());
                        }
                        else if (fPlot == "magnetic_gradient_z") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Z());
                        }
                        else if (fPlot == "magnetic_gradient_z_abs") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.Z()));
                        }
                        else if (fPlot == "magnetic_gradient_x") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.X());
                        }
                        else if (fPlot == "magnetic_gradient_x_abs") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.X()));
                        }
                        else if (fPlot == "magnetic_gradient_y") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Y());
                        }
                        else if (fPlot == "magnetic_gradient_y_abs") {
                            Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.Y()));
                        }
                        else {
                            vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                                           << "> is not defined in KSROOTMagFieldPainter" << eom;
                        }
                    }
                }
                else {
                    vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                                   << "> is not defined in KSROOTMagFieldPainter" << eom;
                }
            }
        }
    }
    else {
        vismsg(eError) << "axial symmetry must be either true or false!" << eom;
    }

    fMap = Map;

    return;
}

void KSROOTMagFieldPainter::FieldMapX(KSMagneticField* tMagField, double tDeltaZ, double tDeltaR)
{
    double tR_i, tR_j;
    auto* Map = new TH2D("Map", "Map", 2 * fRsteps, -fRmax, fRmax, 2 * fRsteps, -fRmax, fRmax);
    KThreeVector tPosition;
    KThreeVector tPosition_i, tPosition_j, tPosition_z;

    KThreeVector tMagneticField;
    KThreeVector tPotential;
    KThreeMatrix tGradient;
    vismsg(eNormal) << "start calculating <" << fPlot << "> map" << eom;
    for (int i = fRsteps; i >= -fRsteps; i--) {
        vismsg(eNormal) << "map: R Position: " << -(i - fRsteps) << "/" << 2 * fRsteps << reom;
        tR_i = i * tDeltaR;
        for (int j = fRsteps; j >= -fRsteps; j--) {
            tR_j = j * tDeltaR;
            tPosition_z.SetComponents(tR_i, tR_j, fZfix + tDeltaZ);
            if (fYAxis == "y") {
                tPosition.SetComponents(tR_i, tR_j, fZfix);
                tPosition_i.SetComponents(tR_i + tDeltaR, tR_j, fZfix);
                tPosition_j.SetComponents(tR_i, tR_j + tDeltaR, fZfix);
            }
            else
                vismsg(eError)
                    << "Please use x for the X-Axis and y for the Y-Axis. All other combinations are not yet included for the xy FieldMap"
                    << eom;
            //				calculate magnetic field at tPosition, requested by any further calculation
            tMagField->CalculateField(tPosition, 0.0, tMagneticField);
            if (fPlot == "magnetic_field_abs") {
                Map->SetBinContent(fRsteps + i + 1, fRsteps + j + 1, tMagneticField.Magnitude());
            }
            else if (fPlot == "magnetic_field_z") {
                Map->SetBinContent(fRsteps + i + 1, fRsteps + j + 1, tMagneticField.Z());
            }
            else if (fPlot == "magnetic_field_z_abs") {
                Map->SetBinContent(fRsteps + i + 1, fRsteps + j + 1, fabs(tMagneticField.Z()));
            }
            else if (fPlot == "magnetic_field_x") {
                Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.X());
            }
            else if (fPlot == "magnetic_field_x_abs") {
                Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.X()));
            }
            else if (fPlot == "magnetic_field_y") {
                Map->SetBinContent(i + 1, fRsteps + j + 1, tMagneticField.Y());
            }
            else if (fPlot == "magnetic_field_y_abs") {
                Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tMagneticField.Y()));
            }
            else if (fPlot == "magnetic_potential_abs") {
                tMagField->CalculatePotential(tPosition, 0.0, tPotential);
                Map->SetBinContent(i + 1, fRsteps + j + 1, tPotential.Magnitude());
            }
            else if (fPlot.compare(9, 8, "gradient") == 0) {
                if (fGradNumerical == true) {
                    KThreeVector tMagneticField_z, tMagneticField_i, tMagneticField_j;
                    if (fPlot == "magnetic_gradient_z") {
                        tMagField->CalculateField(tPosition_z, 0.0, tMagneticField_z);
                        Double_t tGradient = (tMagneticField_z.Magnitude() - tMagneticField.Magnitude()) / tDeltaZ;
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                    }
                    else if (fPlot == "magnetic_gradient_z_abs") {
                        tMagField->CalculateField(tPosition_z, 0.0, tMagneticField_z);
                        Double_t tGradient =
                            fabs((tMagneticField_z.Magnitude() - tMagneticField.Magnitude()) / tDeltaZ);
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                    }
                    else if (fPlot == "magnetic_gradient_x") {
                        tMagField->CalculateField(tPosition_i, 0.0, tMagneticField_i);
                        Double_t tGradient = (tMagneticField_i.Magnitude() - tMagneticField.Magnitude()) / tDeltaR;
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                    }
                    else if (fPlot == "magnetic_gradient_x_abs") {
                        tMagField->CalculateField(tPosition_i, 0.0, tMagneticField_i);
                        Double_t tGradient =
                            fabs((tMagneticField_i.Magnitude() - tMagneticField.Magnitude()) / tDeltaR);
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                    }
                    else if (fPlot == "magnetic_gradient_y") {
                        tMagField->CalculateField(tPosition_j, 0.0, tMagneticField_j);
                        Double_t tGradient = (tMagneticField_j.Magnitude() - tMagneticField.Magnitude()) / tDeltaR;
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                    }
                    else if (fPlot == "magnetic_gradient_y_abs") {
                        tMagField->CalculateField(tPosition_j, 0.0, tMagneticField_j);
                        Double_t tGradient =
                            fabs((tMagneticField_j.Magnitude() - tMagneticField.Magnitude()) / tDeltaR);
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradient);
                    }
                }
                else if (fGradNumerical == false) {
                    //						calculate gradient matrix
                    tMagField->CalculateGradient(tPosition, 0.0, tGradient);
                    KThreeVector tGradB;
                    tGradB.SetX(tMagneticField.X() * tGradient[0] + tMagneticField.Y() * tGradient[1] +
                                tMagneticField.Z() * tGradient[2]);
                    tGradB.SetY(tMagneticField.X() * tGradient[3] + tMagneticField.Y() * tGradient[4] +
                                tMagneticField.Z() * tGradient[5]);
                    tGradB.SetZ(tMagneticField.X() * tGradient[6] + tMagneticField.Y() * tGradient[7] +
                                tMagneticField.Z() * tGradient[8]);
                    tGradB *= tMagneticField.Magnitude();
                    if (fPlot == "magnetic_gradient_abs") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Magnitude());
                    }
                    else if (fPlot == "magnetic_gradient_z") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Z());
                    }
                    else if (fPlot == "magnetic_gradient_z_abs") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.Z()));
                    }
                    else if (fPlot == "magnetic_gradient_x") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.X());
                    }
                    else if (fPlot == "magnetic_gradient_x_abs") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.X()));
                    }
                    else if (fPlot == "magnetic_gradient_y") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, tGradB.Y());
                    }
                    else if (fPlot == "magnetic_gradient_y_abs") {
                        Map->SetBinContent(i + 1, fRsteps + j + 1, fabs(tGradB.Y()));
                    }
                }
            }
            else {
                vismsg(eError) << "do not know what to plot, plot=<" << fPlot
                               << "> is not defined in KSROOTMagFieldPainter" << eom;
            }
        }
    }
    fMap = Map;

    return;
}

void KSROOTMagFieldPainter::Render()
{
    vismsg(eNormal) << "Getting magnetic field <" << fMagneticFieldName << "> from the toolbox" << eom;
    KSMagneticField* tMagField = getMagneticField(fMagneticFieldName);
    if (tMagField == nullptr)
        vismsg(eError) << "No magnetic Field!" << eom;
    vismsg(eNormal) << "Initialize magnetic field (again)" << eom;
    tMagField->Initialize();

    double tDeltaZ = fabs(fZmax - fZmin) / fZsteps;
    double tDeltaR = fabs(fRmax) / fRsteps;

    if (fXAxis == "z") {
        vismsg(eNormal) << "initializing z field map with root_window_x=" << fXAxis << " and root_window_y=" << fYAxis
                        << eom;
        FieldMapZ(tMagField, tDeltaZ, tDeltaR);
    }
    else if (fXAxis == "x") {
        vismsg(eNormal) << "initializing xy field map with root_window_x=" << fXAxis << " and root_window_y=" << fYAxis
                        << eom;
        FieldMapX(tMagField, tDeltaZ, tDeltaR);
    }
    else
        vismsg(eError) << "accept only x or z for x_axis " << eom;

    return;
}

void KSROOTMagFieldPainter::Display()
{
    //palette settings - completely independent
    const Int_t NRGBs = 6;
    const Int_t NCont = 999;

    Double_t stops[NRGBs] = {0.00, 0.1, 0.34, 0.61, 0.84, 1.00};
    Double_t red[NRGBs] = {0.99, 0.0, 0.00, 0.87, 1.00, 0.51};
    Double_t green[NRGBs] = {0.00, 0.0, 0.81, 1.00, 0.20, 0.00};
    Double_t blue[NRGBs] = {0.99, 0.0, 1.00, 0.12, 0.00, 0.00};
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);

    if (fDisplayEnabled == true) {
        fWindow->GetCanvas()->SetRightMargin(0.15);

        //			gStyle->SetPalette(57);
        if (fDraw == "COLZ")
            gStyle->SetNumberContours(99);
        else {
            if (fPlot.compare(0, 14, "magnetic_field") == 0 && fPlot.compare(fPlot.size() - 3, 3, "abs") == 0) {
                Double_t contours[] = {0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6};
                fMap->SetContour(sizeof(contours) / sizeof(Double_t), contours);
                fMap->GetZaxis()->SetRangeUser(0.000001, 6);
                fUseLogZ = true;
            }
            else {
                gStyle->SetNumberContours(99);
            }
        }

        if (fPlot == "magnetic_field_abs")
            fMap->SetZTitle("abs(Magnetic Field) (T)");
        else if (fPlot == "magnetic_field_z")
            fMap->SetZTitle("Magnetic Field_z (T)");
        else if (fPlot == "magnetic_field_z_abs")
            fMap->SetZTitle("abs(Magnetic Field_z) (T)");
        else if (fPlot == "magnetic_field_x")
            fMap->SetZTitle("Magnetic Field_x (T)");
        else if (fPlot == "magnetic_field_x_abs")
            fMap->SetZTitle("abs(Magnetic Field_x) (T)");
        else if (fPlot == "magnetic_field_y")
            fMap->SetZTitle("Magnetic Field_y (T)");
        else if (fPlot == "magnetic_field_y_abs")
            fMap->SetZTitle("abs(Magnetic Field_y) (T)");
        else if (fPlot == "magnetic_gradient")
            fMap->SetZTitle("Magnetic Gradient (T/m)");
        else if (fPlot == "magnetic_gradient_z")
            fMap->SetZTitle("Magnetic Gradient_z (T/m)");
        else if (fPlot == "magnetic_gradient_z_abs")
            fMap->SetZTitle("abs(Magnetic Gradient_z) (T/m)");
        else if (fPlot == "magnetic_gradient_x")
            fMap->SetZTitle("Magnetic Gradient_x (T/m)");
        else if (fPlot == "magnetic_gradient_x_abs")
            fMap->SetZTitle("abs(Magnetic Gradient_x) (T/m)");
        else if (fPlot == "magnetic_gradient_y")
            fMap->SetZTitle("Magnetic Gradient_y (T/m)");
        else if (fPlot == "magnetic_gradient_y_abs")
            fMap->SetZTitle("abs(Magnetic Gradient_y) (T/m)");
        else if (fPlot == "magnetic_potential")
            fMap->SetZTitle("Magnetic Potential (T*m)");

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
        if (fUseLogZ == true)
            fWindow->GetCanvas()->SetLogz();

        fMap->GetZaxis()->SetTitleOffset(1.4);
        fMap->SetStats(false);
        fMap->SetTitle("");
        fMap->Draw(fDraw.c_str());
    }
    return;
}

void KSROOTMagFieldPainter::Write() {}

double KSROOTMagFieldPainter::GetXMin()
{
    double tMin(std::numeric_limits<double>::max());
    return tMin;
}
double KSROOTMagFieldPainter::GetXMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    return tMax;
}

double KSROOTMagFieldPainter::GetYMin()
{
    double tMin(std::numeric_limits<double>::max());
    return tMin;
}
double KSROOTMagFieldPainter::GetYMax()
{
    double tMax(-1.0 * std::numeric_limits<double>::max());
    return tMax;
}

std::string KSROOTMagFieldPainter::GetXAxisLabel()
{
    return fXAxis;
}

std::string KSROOTMagFieldPainter::GetYAxisLabel()
{
    return fYAxis;
}

}  // namespace Kassiopeia
