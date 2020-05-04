/*
 * KEMElectricFieldPointsRootFile.hh
 *
 *  Created on: 04.05.2015
 *      Author: gosda
 */

#ifndef KEMELECTRICFIELDPOINTSROOTFILE_HH_
#define KEMELECTRICFIELDPOINTSROOTFILE_HH_

#include "KThreeVector_KEMField.hh"
#include "TFile.h"
#include "TTree.h"

#include <string>

namespace KEMField
{

class KEMElectricFieldPointsRootFile
{
  public:
    KEMElectricFieldPointsRootFile(std::string fullPath);
    ~KEMElectricFieldPointsRootFile();

    void append(KPosition position, KDirection eField, double potential);
    void Write();

  private:
    TFile fFile;
    TTree fTree;
    KPosition fPosition;
    double fPotential;
    KDirection fElectricField;
    double fElectricFieldAbs;
};

}  // namespace KEMField

#endif /* KEMELECTRICFIELDPOINTSROOTFILE_HH_ */
