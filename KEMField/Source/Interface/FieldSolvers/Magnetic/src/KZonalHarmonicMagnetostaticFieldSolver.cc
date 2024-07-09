/*
 * KZonalHarmonicMagnetostaticFieldSolver.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KZonalHarmonicMagnetostaticFieldSolver.hh"

#include "KEMCoreMessage.hh"
#include "KEMFileInterface.hh"
#include "KMD5HashGenerator.hh"

using namespace std;

namespace KEMField
{

KZonalHarmonicMagnetostaticFieldSolver::KZonalHarmonicMagnetostaticFieldSolver() :
    fZHContainer(nullptr),
    fZonalHarmonicFieldSolver(nullptr)
{
    fParameters = std::make_shared<KZonalHarmonicParameters>();
}

KZonalHarmonicMagnetostaticFieldSolver::~KZonalHarmonicMagnetostaticFieldSolver()
{
    delete fZHContainer;
    delete fZonalHarmonicFieldSolver;
}

void KZonalHarmonicMagnetostaticFieldSolver::InitializeCore(KElectromagnetContainer& container)
{
    // compute hash of the solved geometry
    KMD5HashGenerator solutionHashGenerator;
    string solutionHash = solutionHashGenerator.GenerateHash(container);

    kem_cout_debug("<shape+boundary+solution> hash is <" << solutionHash << ">" << eom);

    // compute hash of the parameter values on the bare geometry
    KMD5HashGenerator parameterHashGenerator;
    string parameterHash = parameterHashGenerator.GenerateHash(*fParameters);

    kem_cout_debug("<parameter> hash is <" << parameterHash << ">" << eom);

    // create label set for zh container object
    string zhContainerBase(KZonalHarmonicContainer<KMagnetostaticBasis>::Name());
    string zhContainerName = zhContainerBase + string("_") + solutionHash + string("_") + parameterHash;
    vector<string> zhContainerLabels;
    zhContainerLabels.push_back(zhContainerBase);
    zhContainerLabels.push_back(solutionHash);
    zhContainerLabels.push_back(parameterHash);

    //auto* tParametersCopy = new KZonalHarmonicParameters;
    //*tParametersCopy = *fParameters;

    fZHContainer = new KZonalHarmonicContainer<KMagnetostaticBasis>(container, fParameters);

    bool containerFound = false;
    string containerFilename;

    KEMFileInterface::GetInstance()->FindByLabels(*fZHContainer,
                                                  zhContainerLabels,
                                                  0,
                                                  containerFound,
                                                  containerFilename);

    if (containerFound == true) {
        kem_cout() << "zonal harmonic container found in file <" << containerFilename << ">" << eom;
    }
    else {
        kem_cout(eInfo) << "no zonal harmonic container found." << eom;

        fZHContainer->ComputeCoefficients();

        KEMFileInterface::GetInstance()->Write(*fZHContainer, zhContainerName, zhContainerLabels);
    }

    fZonalHarmonicFieldSolver = new KZonalHarmonicFieldSolver<KMagnetostaticBasis>(*fZHContainer, fIntegrator);
    fZonalHarmonicFieldSolver->Initialize();

    return;
}

void KZonalHarmonicMagnetostaticFieldSolver::DeinitializeCore()
{
    const auto& C = fZonalHarmonicFieldSolver->GetCentralExecutionCount();
    const auto& R = fZonalHarmonicFieldSolver->GetRemoteExecutionCount();
    const auto& D = fZonalHarmonicFieldSolver->GetDirectExecutionCount();
    const auto& T = fZonalHarmonicFieldSolver->GetTotalExecutionCount();
    kem_cout(eNormal) << "Magnetic ZH solver execution counts:" << ret
                      << "central: " << C << " (" << std::floor(100.*C/T) << "%)" << ret
                      << "remote:  " << R << " (" << std::floor(100.*R/T) << "%)" << ret
                      << "direct:  " << D << " (" << std::floor(100.*D/T) << "%)" << eom;
}

KFieldVector KZonalHarmonicMagnetostaticFieldSolver::MagneticPotentialCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->VectorPotential(P);
}

KFieldVector KZonalHarmonicMagnetostaticFieldSolver::MagneticFieldCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->MagneticField(P);
}

KGradient KZonalHarmonicMagnetostaticFieldSolver::MagneticGradientCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->MagneticFieldGradient(P);
}

std::pair<KFieldVector, KGradient>
KZonalHarmonicMagnetostaticFieldSolver::MagneticFieldAndGradientCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->MagneticFieldAndGradient(P);
}

bool KZonalHarmonicMagnetostaticFieldSolver::UseCentralExpansion(const KPosition& P)
{
    return fZonalHarmonicFieldSolver->CentralExpansion(P);
}

bool KZonalHarmonicMagnetostaticFieldSolver::UseRemoteExpansion(const KPosition& P)
{
    return fZonalHarmonicFieldSolver->RemoteExpansion(P);
}

const KZonalHarmonicContainer<KMagnetostaticBasis>* KZonalHarmonicMagnetostaticFieldSolver::GetContainer() const
{
    return fZHContainer;
}

} /* namespace KEMField */
