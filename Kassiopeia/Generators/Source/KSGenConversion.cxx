/*
 * KKSGenConversion.cxx
 *
 *  Created on: 20.04.2010
 *      Author: mertens/wandkowsky
 */

#include "KSGenConversion.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"

using namespace std;
using namespace katrin;

namespace Kassiopeia
{

KSGenConversion::KSGenConversion() : fForceCreation(false), fIsotope(0) {}
KSGenConversion::~KSGenConversion()
{
    delete fDataFile;
}

bool KSGenConversion::Initialize(int isotope)
{
    SetIsotope(isotope);

    if (ReadData() == false) {
        genmsg(eError) << "KSGenConversion::KSGenConversion" << ret;
        genmsg << "unknown ERROR reading file. quit." << eom;
        return false;
    }
    else {
        return false;
    }
}

void KSGenConversion::CreateCE(vector<int>& vacancy, vector<double>& energy)
{
    //for the case of krypton the probability of producing a CE is very high, there are two transitions in both a CE can be produced
    //the for loop is running over all possible transitions
    //for each transition it is diced whether an conversion electron is produced in that transition
    //many CE can be produced
    //however, since there is no normalization of each transition, it can rarely happen that no CE is produced
    DoDoubleConversion = 0;

    if (fForceCreation == false) {
        for (unsigned int j = 0; j < fConvProb.size(); j++) {
            double myRandNr = KRandom::GetInstance().Uniform();
            unsigned int i = 0;
            while (myRandNr > 0 && i < fConvProb.at(j).size()) {
                myRandNr -= fConvProb.at(j).at(i);
                i++;
            }
            if (myRandNr < 0) {
                DoDoubleConversion = fDoubleConv.at(j).at(i - 1);
                vacancy.push_back(fShell.at(j).at(i - 1));
                energy.push_back(fConvE.at(j).at(i - 1));
                break;
            }
        }

        if (DoDoubleConversion != 0) {
            double myRandNr = KRandom::GetInstance().Uniform();
            unsigned int i = 0;
            while (myRandNr > 0 && i < fConvProb.at(DoDoubleConversion - 1).size()) {
                myRandNr -= fConvProb.at(DoDoubleConversion - 1).at(i);
                i++;
            }
            if (myRandNr < 0) {
                genmsg(eDebug) << "Double Conversion Happened!" << eom;
                vacancy.push_back(fShell.at(DoDoubleConversion - 1).at(i - 1));
                energy.push_back(fConvE.at(DoDoubleConversion - 1).at(i - 1));
            }
        }
    }

    if (fForceCreation == true) {
        //for the case of radon (for instance) the probability for producing a CE is very low.
        //To speed up the simulation, all transition probabilities can be normalized to 1, to force the creation of a CE.
        //No hierarchy is considered in that case
        //In this case a single CE must be produced
        double myRandNr = KRandom::GetInstance().Uniform();
        unsigned int i = 0;
        unsigned int j = 0;
        while (myRandNr > 0 && j < fConvProbNorm.size()) {
            i = 0;
            while (myRandNr > 0 && i < fConvProbNorm.at(j).size()) {

                myRandNr -= fConvProbNorm.at(j).at(i);
                i++;
            }
            j++;
        }
        if (myRandNr < 0) {
            vacancy.push_back(fShell.at(j - 1).at(i - 1));
            energy.push_back(fConvE.at(j - 1).at(i - 1));
        }
        else {
            genmsg(eError) << "KSGenConversion::CreateCE" << ret;
            genmsg << ": no conversionVacancy." << eom;
        }
    }
}

bool KSGenConversion::ReadData()
{

    double prob, probNorm, energy, probTotal = 0;
    int shell, doubleconv, transition;

    if (fIsotope == 219) {
        fDataFile = KTextFile::CreateDataTextFile("ConversionRn219.dat");
    }
    else if (fIsotope == 220) {
        fDataFile = KTextFile::CreateDataTextFile("ConversionRn220.dat");
    }
    else if (fIsotope == 83) {
        fDataFile = KTextFile::CreateDataTextFile("ConversionKr83.dat");
    }
    else if (fIsotope == 210) {
        fDataFile = KTextFile::CreateDataTextFile("ConversionPb210.dat");
    }
    else {
        genmsg(eError) << "KSGenConversion::ReadData" << ret;
        genmsg << "Isotope " << fIsotope << " not supported by conversion process!" << eom;
    }

    if (fDataFile->Open(KFile::eRead) == true) {

        fstream& inputfile = *(fDataFile->File());

        int oldtransition = -1;

        while (!inputfile.eof()) {

            char c = inputfile.peek();
            if (c >= '0' && c <= '9') {
                inputfile >> shell >> transition >> prob >> energy >> doubleconv;
                probTotal = probTotal + prob;

                if (transition != oldtransition) {
                    vector<int> tempvec;
                    fShell.push_back(tempvec);
                    vector<int> tempvec1;
                    fDoubleConv.push_back(tempvec1);
                    vector<double> tempvec2;
                    fConvE.push_back(tempvec2);
                    vector<double> tempvec3;
                    fConvProb.push_back(tempvec3);
                    oldtransition = transition;
                }

                fShell.back().push_back(shell);
                fDoubleConv.back().push_back(doubleconv);
                fConvE.back().push_back(energy);
                fConvProb.back().push_back(prob);
            }
            else {
                char dump[200];
                inputfile.getline(dump, 200);
                genmsg_debug("KSGenConversion::ReadData " << ret);
                genmsg_debug("dumping " << dump << " because " << c << " is not a number" << eom);
                continue;
            }
        }
    }
    else {
        genmsg(eError) << "KSGenConversion::ReadData" << ret;
        genmsg(eError) << "could not open conversion data file" << eom;
    }
    fDataFile->Close();

    double total = 0;

    for (auto& j : fConvProb) {
        vector<double> tempvec4;
        fConvProbNorm.push_back(tempvec4);
        for (double i : j) {
            probNorm = i / probTotal;
            fConvProbNorm.back().push_back(probNorm);
            total = total + probNorm;
        }
    }

    return true;
}

}  //namespace Kassiopeia
