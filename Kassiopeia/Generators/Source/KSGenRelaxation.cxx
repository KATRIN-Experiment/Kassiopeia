/*
 * KSGenRelaxation.cxx
 *
 *  Created on: 29.03.2010
 *      Author: renschler, mertens
 */

#include "KRandom.h"
using katrin::KRandom;
#include "KSGenRelaxation.h"
#include "KSGeneratorsMessage.h"

using namespace std;

namespace Kassiopeia
{

KSGenRelaxation::KSGenRelaxation() : fVacancies(new std::vector<unsigned int>), fIsotope(0) {}

KSGenRelaxation::~KSGenRelaxation()
{
    delete fVacancies;
    delete fDataFile;
}

bool KSGenRelaxation::Initialize(int isotope)
{
    SetIsotope(isotope);

    if (ReadData() == false) {
        genmsg(eError) << "KSGenRelaxation::KSGenRelaxation" << ret;
        genmsg << "unknown ERROR reading file. quit." << eom;
        return false;
    }
    else {
        return true;
    }
}

bool KSGenRelaxation::ReadData()
{

    genmsg_debug("KSGenRelaxation::ReadTables");

    string textline;
    unsigned int numberOfElectronStates;
    double four = 0;
    unsigned int one = 0, three = 0;
    string two;

    if (fIsotope == 219 || fIsotope == 220) {
        fDataFile = katrin::CreateDataTextFile("RelaxationPo.dat");
    }
    else if (fIsotope == 83) {
        fDataFile = katrin::CreateDataTextFile("RelaxationKr.dat");
    }
    else if (fIsotope == 210) {
        fDataFile = katrin::CreateDataTextFile("RelaxationBi.dat");
    }
    else {
        genmsg(eError) << "KSGenConversion::ReadData" << ret;
        genmsg << "isotope " << fIsotope << " not supported by relaxation process!" << eom;
    }

    if (fDataFile->Open(katrin::KFile::eRead) == true) {

        fstream& inputfile = *(fDataFile->File());

        while (!inputfile.eof()) {

            char c = inputfile.peek();
            if (c >= '0' && c < '9') {
                inputfile >> numberOfElectronStates;

                for (unsigned int i = 0; i < numberOfElectronStates; i++) {
                    inputfile >> one >> two >> three >> four;
                    if (inputfile.eof() == false) {
                        fshellEnergies.insert(pair<unsigned int, double>(one, four));
                    }
                }
                while (inputfile.eof() == false) {
                    unsigned int vacOne, vacTwo, vacThree;
                    double probability, energy;
                    inputfile >> vacOne >> vacTwo >> vacThree >> probability >> energy;

                    line myLine;
                    myLine.vacOne = vacOne;
                    myLine.vacTwo = vacTwo;
                    myLine.vacThree = vacThree;
                    myLine.probability = probability;
                    myLine.energy = energy;

                    fTransProp.push_back(myLine);
                }
            }
            else {
                char dump[200];
                inputfile.getline(dump, 200);
                genmsg_debug("KSGenRelaxation::ReadTables " << ret);
                genmsg_debug("dumping " << dump << " because " << c << " is not a number" << eom);
                continue;
            }
        }
    }
    fDataFile->Close();
    return true;
}

void KSGenRelaxation::RelaxVacancy(unsigned int shell)
{
    double myRandNr = KRandom::GetInstance().Uniform();

    int i = 0;
    while (myRandNr > 0.) {
        if (fTransProp.at(i).vacOne == shell) {
            myRandNr -= fTransProp.at(i).probability;
        }
        i++;
    }
    if (fTransProp.at(i - 1).vacThree == 0) {
        //fluorescence
        fFluorescenceEnergies.push_back(fTransProp.at(i - 1).energy);
        fVacancies->push_back(fTransProp.at(i - 1).vacTwo);
    }
    else {
        //auger
        fAugerEnergies.push_back(fTransProp.at(i - 1).energy);
        fVacancies->push_back(fTransProp.at(i - 1).vacTwo);
        fVacancies->push_back(fTransProp.at(i - 1).vacThree);
    }
}

void KSGenRelaxation::Relax()
{
    while (fVacancies->size() > 0) {
        if (fVacancies->back() < fTransProp.back().vacOne + 1) {
            unsigned int shell = fVacancies->back();
            fVacancies->pop_back();
            this->RelaxVacancy(shell);
        }
        else {
            fVacancies->pop_back();
        }
    }
}

void KSGenRelaxation::Relax(unsigned int vacancy)
{
    fVacancies->push_back(vacancy);
    this->Relax();
}

}  //namespace Kassiopeia
