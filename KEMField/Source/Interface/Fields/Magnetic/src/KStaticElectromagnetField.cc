/*
 * KStaticElectroMagnetField.cc
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#include "KStaticElectromagnetField.hh"
#include "KEMSimpleException.hh"
#include "KEMFileInterface.hh"

namespace KEMField {

KStaticElectromagnetField::KStaticElectromagnetField() :
            fContainer( NULL ),
            fFieldSolver( NULL ),
            fFile(),
            fDirectory (KEMFileInterface::GetInstance()->ActiveDirectory() )
{
    fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
    fFile = fFile.substr( fFile.find_last_of( "/" ) + 1, std::string::npos );
}

KStaticElectromagnetField::~KStaticElectromagnetField()
{
}

KThreeVector KStaticElectromagnetField::MagneticPotentialCore(
        const KPosition& aSamplePoint) const {
    return fFieldSolver->MagneticPotential(aSamplePoint);
}

KThreeVector KStaticElectromagnetField::MagneticFieldCore(
        const KPosition& aSamplePoint) const {
    return fFieldSolver->MagneticField(aSamplePoint);
}

KGradient KStaticElectromagnetField::MagneticGradientCore(
        const KPosition& aSamplePoint) const {
    return fFieldSolver->MagneticGradient(aSamplePoint);
}

std::pair<KThreeVector, KGradient>
KStaticElectromagnetField::MagneticFieldAndGradientCore(const KPosition& P) const
{
    return fFieldSolver->MagneticFieldAndGradient(P);
}

void KStaticElectromagnetField::SetDirectory(const std::string& aDirectory) {
    fDirectory = aDirectory;
}

void KStaticElectromagnetField::SetFile(const std::string& aFile) {
    fFile = aFile;
}

void KStaticElectromagnetField::SetFieldSolver(KSmartPointer<KMagneticFieldSolver> solver) {
    fFieldSolver = solver;
}

KSmartPointer<KMagneticFieldSolver> KStaticElectromagnetField::GetFieldSolver() {
    return fFieldSolver;
}

void KStaticElectromagnetField::SetContainer(
        KSmartPointer<KElectromagnetContainer> aContainer) {
    fContainer = aContainer;
}

KSmartPointer<KElectromagnetContainer> KStaticElectromagnetField::GetContainer() const {
    return fContainer;
}

void KStaticElectromagnetField::InitializeCore() {
    CheckSolverExistance();

    KEMFileInterface::GetInstance()->ActiveDirectory( fDirectory );
    KEMFileInterface::GetInstance()->ActiveFile( KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFile );

    fFieldSolver->Initialize(*fContainer);
}

void KStaticElectromagnetField::CheckSolverExistance() const {
    if(!fFieldSolver )
        throw KEMSimpleException("Initializing aborted: no field solver!");
}

} /* namespace KEMField */
