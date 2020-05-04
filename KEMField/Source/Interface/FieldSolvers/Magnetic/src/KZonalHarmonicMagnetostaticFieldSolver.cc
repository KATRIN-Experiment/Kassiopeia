/*
 * KZonalHarmonicMagnetostaticFieldSolver.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KZonalHarmonicMagnetostaticFieldSolver.hh"

#include "KEMCout.hh"
#include "KEMFileInterface.hh"
#include "KMD5HashGenerator.hh"

using namespace std;

namespace KEMField
{

KZonalHarmonicMagnetostaticFieldSolver::KZonalHarmonicMagnetostaticFieldSolver() :
    fZHContainer(nullptr),
    fZonalHarmonicFieldSolver(nullptr)
{
    fParameters = new KZonalHarmonicParameters();
}
KZonalHarmonicMagnetostaticFieldSolver::~KZonalHarmonicMagnetostaticFieldSolver()
{
    delete fZHContainer;
    delete fZonalHarmonicFieldSolver;
    delete fParameters;
}
void KZonalHarmonicMagnetostaticFieldSolver::InitializeCore(KElectromagnetContainer& container)
{
    // compute hash of the solved geometry
    KMD5HashGenerator solutionHashGenerator;
    string solutionHash = solutionHashGenerator.GenerateHash(container);

    //KEMField::cout << "<shape+boundary+solution> hash is <" << solutionHash << ">" << endl;

    // compute hash of the parameter values on the bare geometry
    KMD5HashGenerator parameterHashGenerator;
    string parameterHash = parameterHashGenerator.GenerateHash(*fParameters);

    //KEMField::cout << "<parameter> hash is <" << parameterHash << ">" << endl;

    // create label set for zh container object
    string zhContainerBase(KZonalHarmonicContainer<KMagnetostaticBasis>::Name());
    string zhContainerName = zhContainerBase + string("_") + solutionHash + string("_") + parameterHash;
    vector<string> zhContainerLabels;
    zhContainerLabels.push_back(zhContainerBase);
    zhContainerLabels.push_back(solutionHash);
    zhContainerLabels.push_back(parameterHash);

    auto* tParametersCopy = new KZonalHarmonicParameters;
    *tParametersCopy = *fParameters;

    fZHContainer = new KZonalHarmonicContainer<KMagnetostaticBasis>(container, tParametersCopy);

    bool containerFound = false;

    KEMFileInterface::GetInstance()->FindByLabels(*fZHContainer, zhContainerLabels, 0, containerFound);

    if (containerFound == true) {
        KEMField::cout << "zonal harmonic container found." << endl;
    }
    else {
        //KEMField::cout << "no zonal harmonic container found." << endl;

        fZHContainer->ComputeCoefficients();

        KEMFileInterface::GetInstance()->Write(*fZHContainer, zhContainerName, zhContainerLabels);
    }

    fZonalHarmonicFieldSolver = new KZonalHarmonicFieldSolver<KMagnetostaticBasis>(*fZHContainer, fIntegrator);
    fZonalHarmonicFieldSolver->Initialize();

    return;
}

KThreeVector KZonalHarmonicMagnetostaticFieldSolver::MagneticPotentialCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->VectorPotential(P);
}

KThreeVector KZonalHarmonicMagnetostaticFieldSolver::MagneticFieldCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->MagneticField(P);
}

KGradient KZonalHarmonicMagnetostaticFieldSolver::MagneticGradientCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->MagneticFieldGradient(P);
}

std::pair<KThreeVector, KGradient>
KZonalHarmonicMagnetostaticFieldSolver::MagneticFieldAndGradientCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->MagneticFieldAndGradient(P);
}

} /* namespace KEMField */
