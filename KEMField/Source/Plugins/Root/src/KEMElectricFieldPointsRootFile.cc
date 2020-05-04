/*
 * KEMElectricFieldPointsRootFile.cc
 *
 *  Created on: 04.05.2015
 *      Author: gosda
 */

#include "KEMElectricFieldPointsRootFile.hh"

namespace KEMField
{

KEMElectricFieldPointsRootFile::KEMElectricFieldPointsRootFile(std::string fullPath) :
    fFile(fullPath.c_str(), "RECREATE"),
    fTree("field_values", "field_values")
{
    fTree.Branch("posX", &fPosition.X(), "posX/D");
    fTree.Branch("posY", &fPosition.Y(), "posY/D");
    fTree.Branch("posZ", &fPosition.Z(), "posZ/D");
    fTree.Branch("ElPot", &fPotential, "ElPot/D");
    fTree.Branch("ElFieldX", &fElectricField.X(), "ElFieldX/D");
    fTree.Branch("ElFieldY", &fElectricField.Y(), "ElFieldY/D");
    fTree.Branch("ElFieldZ", &fElectricField.Z(), "ElFieldZ/D");
    fTree.Branch("ElFieldAbs", &fElectricFieldAbs, "ElFieldAbs/D");
}

KEMElectricFieldPointsRootFile::~KEMElectricFieldPointsRootFile()
{
    Write();
}

void KEMElectricFieldPointsRootFile::append(KPosition position, KDirection eField, double potential)
{
    fPosition = position;
    fElectricField = eField;
    fElectricFieldAbs = fElectricField.Magnitude();
    fPotential = potential;
    fTree.Fill();
}

void KEMElectricFieldPointsRootFile::Write()
{
    fFile.Write();
}

}  // namespace KEMField
