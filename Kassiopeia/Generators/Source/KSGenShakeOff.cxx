/*
 * KKSGenShakeOff.cxx
 *
 *  Created on: 31.03.2010
 *      Author: mertens
 */

#include "KSGenShakeOff.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"

#include <cmath>

using namespace std;
using namespace katrin;

namespace Kassiopeia
{

KSGenShakeOff::KSGenShakeOff()
{
    if (ReadData() == false) {
        genmsg(eError) << "KSGenShakeOff::KSGenShakeOff" << ret;
        genmsg << "unknown ERROR reading file. quit." << eom;
    }
}

KSGenShakeOff::~KSGenShakeOff()
{
    delete fDataFile;
}

void KSGenShakeOff::CreateSO(vector<int>& vacancy, vector<double>& energy)
{
    double myRandNr = KRandom::GetInstance().Uniform();
    unsigned int i = 0;
    while (myRandNr > 0. && i < fSoProb.size()) {
        if (fForceCreation == false) {
            myRandNr -= fSoProb.at(i);
        }
        else {
            myRandNr -= fSoProbNorm.at(i);
        }
        i++;
    }

    if (myRandNr < 0) {
        vacancy.push_back(fShell.at(i - 1));
        energy.push_back(DiceEnergy(fBindE.at(i - 1), fShell.at(i - 1)));
    }

    return;
}

double KSGenShakeOff::DiceEnergy(double bindingEnergy, int vacancy)
{
    double maxEnergy;

    if (vacancy == 1) {
        maxEnergy = 100000;
        //cout <<"K-shell vacancy!"<<endl;
    }
    else {
        if (vacancy > 1 && vacancy < 5) {
            maxEnergy = 90000;
            //cout <<"L-shell vacancy!"<<endl;
        }
        else {
            maxEnergy = 25000;
            //cout <<"M-shell vacancy!"<<endl;
        }
    }
    double myRandNrx = KRandom::GetInstance().Uniform() * maxEnergy;
    double myRandNry = KRandom::GetInstance().Uniform();

    while (myRandNry > pow(bindingEnergy / (bindingEnergy + myRandNrx), 8)) {
        myRandNrx = KRandom::GetInstance().Uniform() * maxEnergy;
        myRandNry = KRandom::GetInstance().Uniform();
    }
    //cout <<"Shell-energy: "<<myRandNrx<<endl;
    return myRandNrx;
}

bool KSGenShakeOff::ReadData()
{

    double shell, /*subshell,*/ prob, probNorm, energy, probTotal = 0;

    fDataFile = katrin::CreateDataTextFile("ShakeOffRn.dat");

    if (fDataFile->Open(katrin::KFile::eRead) == true) {

        fstream& inputfile = *(fDataFile->File());

        while (!inputfile.eof()) {

            char c = inputfile.peek();
            if (c >= '0' && c <= '9') {
                inputfile >> shell >> prob >> energy;
                probTotal += prob * 1.;
                fShell.push_back(shell);
                fBindE.push_back(energy);
                fSoProb.push_back(prob * 1.);
            }
            else {
                char dump[200];
                inputfile.getline(dump, 200);
                genmsg_debug("KSGenShakeOff::ReadData " << ret);
                genmsg_debug("dumping " << dump << " because " << c << " is not a number" << eom);
                continue;
            }
        }
    }
    fDataFile->Close();

    for (unsigned int i = 0; i < fSoProb.size(); i++) {
        probNorm = fSoProb.at(i) / probTotal;
        fSoProbNorm.push_back(probNorm);
    }
    return true;
}

}  //namespace Kassiopeia
