/*
 * KStaticElectroMagnetField.cc
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#include "KStaticElectromagnetField.hh"

#include "KEMFileInterface.hh"
#include "KEMSimpleException.hh"

namespace KEMField
{

KStaticElectromagnetField::KStaticElectromagnetField() :
    fContainer(nullptr),
    fFieldSolver(nullptr),
    fDirectory(KEMFileInterface::GetInstance()->ActiveDirectory())
{
    fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
    fFile = fFile.substr(fFile.find_last_of('/') + 1, std::string::npos);
}

KStaticElectromagnetField::~KStaticElectromagnetField() = default;

KFieldVector KStaticElectromagnetField::MagneticPotentialCore(const KPosition& aSamplePoint) const
{
    return fFieldSolver->MagneticPotential(aSamplePoint);
}

KFieldVector KStaticElectromagnetField::MagneticFieldCore(const KPosition& aSamplePoint) const
{
    return fFieldSolver->MagneticField(aSamplePoint);
}

KGradient KStaticElectromagnetField::MagneticGradientCore(const KPosition& aSamplePoint) const
{
    return fFieldSolver->MagneticGradient(aSamplePoint);
}

std::pair<KFieldVector, KGradient> KStaticElectromagnetField::MagneticFieldAndGradientCore(const KPosition& P) const
{
    return fFieldSolver->MagneticFieldAndGradient(P);
}

void KStaticElectromagnetField::SetDirectory(const std::string& aDirectory)
{
    fDirectory = aDirectory;
}

std::string KStaticElectromagnetField::GetDirectory() const
{
    return fDirectory;
}

void KStaticElectromagnetField::SetFile(const std::string& aFile)
{
    fFile = aFile;
}

std::string KStaticElectromagnetField::GetFile() const
{
    return fFile;
}

void KStaticElectromagnetField::SetFieldSolver(const std::shared_ptr<KMagneticFieldSolver>& solver)
{
    fFieldSolver = solver;
}

std::shared_ptr<KMagneticFieldSolver> KStaticElectromagnetField::GetFieldSolver()
{
    return fFieldSolver;
}

void KStaticElectromagnetField::SetContainer(const std::shared_ptr<KElectromagnetContainer>& aContainer)
{
    fContainer = aContainer;
}

std::shared_ptr<KElectromagnetContainer> KStaticElectromagnetField::GetContainer() const
{
    return fContainer;
}

void KStaticElectromagnetField::InitializeCore()
{
    if (fContainer->empty()) {
        kem_cout(eError) << "ERROR: Electromagnet solver got no current elements (did you forget to define magnets?)" << eom;
    }

    CheckSolverExistance();

    KEMFileInterface::GetInstance()->ActiveDirectory(fDirectory);
    KEMFileInterface::GetInstance()->ActiveFile(KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFile);

    fFieldSolver->Initialize(*fContainer);
}

void KStaticElectromagnetField::CheckSolverExistance() const
{
    if (!fFieldSolver)
        throw KEMSimpleException("Initializing aborted: no field solver!");
}

} /* namespace KEMField */
