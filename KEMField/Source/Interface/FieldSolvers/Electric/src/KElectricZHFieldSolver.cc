/*
 * KElectricZHFieldSolver.cc
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#include "KElectricZHFieldSolver.hh"

#include "KEMCoreMessage.hh"
#include "KEMFileInterface.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticZonalHarmonicFieldSolver.hh"
#include "KMD5HashGenerator.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"

#include <string>
#include <vector>

//#include "KElectrostaticBasis.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif

#ifdef KEMFIELD_USE_MPI
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (KEMField::KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (true)
#endif
#endif

using std::string;
using std::vector;

namespace KEMField
{

KElectricZHFieldSolver::KElectricZHFieldSolver() :
    fIntegrator(KEBIFactory::MakeDefault()),
    fZHContainer(nullptr),
    fZonalHarmonicFieldSolver(nullptr)
{
    fParameters = std::make_shared<KZonalHarmonicParameters>();
}

KElectricZHFieldSolver::~KElectricZHFieldSolver()
{
    delete fZHContainer;
    delete fZonalHarmonicFieldSolver;
}

void KElectricZHFieldSolver::InitializeCore(KSurfaceContainer& container)
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
    string zhContainerBase(KZonalHarmonicContainer<KElectrostaticBasis>::Name());
    string zhContainerName = zhContainerBase + string("_") + solutionHash + string("_") + parameterHash;
    vector<string> zhContainerLabels;
    zhContainerLabels.push_back(zhContainerBase);
    zhContainerLabels.push_back(solutionHash);
    zhContainerLabels.push_back(parameterHash);

    // BUG: KZonalHarmonicContainer might delete fParameters
    fZHContainer = new KZonalHarmonicContainer<KElectrostaticBasis>(container, fParameters);

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

        MPI_SINGLE_PROCESS
        {
            KEMFileInterface::GetInstance()->Write(*fZHContainer, zhContainerName, zhContainerLabels);
        }
    }

    fIntegrator = fIntegratorPolicy.CreateIntegrator();
    fZonalHarmonicFieldSolver = new KZonalHarmonicFieldSolver<KElectrostaticBasis>(*fZHContainer, fIntegrator);
    fZonalHarmonicFieldSolver->Initialize();

    return;
}

void KElectricZHFieldSolver::DeinitializeCore()
{
    const auto& C = fZonalHarmonicFieldSolver->GetCentralExecutionCount();
    const auto& R = fZonalHarmonicFieldSolver->GetRemoteExecutionCount();
    const auto& D = fZonalHarmonicFieldSolver->GetDirectExecutionCount();
    const auto& T = fZonalHarmonicFieldSolver->GetTotalExecutionCount();
    kem_cout(eNormal) << "Electric ZH solver execution counts:" << ret
                      << "central: " << C << " (" << std::floor(100.*C/T) << "%)" << ret
                      << "remote:  " << R << " (" << std::floor(100.*R/T) << "%)" << ret
                      << "direct:  " << D << " (" << std::floor(100.*D/T) << "%)" << eom;
}

double KElectricZHFieldSolver::PotentialCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->Potential(P);
}

KFieldVector KElectricZHFieldSolver::ElectricFieldCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->ElectricField(P);
}

std::pair<KFieldVector, double> KElectricZHFieldSolver::ElectricFieldAndPotentialCore(const KPosition& P) const
{
    return fZonalHarmonicFieldSolver->ElectricFieldAndPotential(P);
}

bool KElectricZHFieldSolver::UseCentralExpansion(const KPosition& P)
{
    return fZonalHarmonicFieldSolver->CentralExpansion(P);
}

bool KElectricZHFieldSolver::UseRemoteExpansion(const KPosition& P)
{
    return fZonalHarmonicFieldSolver->RemoteExpansion(P);
}

const KZonalHarmonicContainer<KElectrostaticBasis>* KElectricZHFieldSolver::GetContainer() const
{
    return fZHContainer;
}

} /* namespace KEMField */
