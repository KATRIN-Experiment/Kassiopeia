/*
 * KMagnetostaticBoundaryField.cc
 *
 *  Created on: 07.05.2025
 *      Author: pslocum
 */
#include "KMagnetostaticBoundaryField.hh"

#include "KEMSimpleException.hh"

using namespace std;

namespace KEMField
{

KMagnetostaticBoundaryField::KMagnetostaticBoundaryField() :
    fChargeDensitySolver(nullptr),
    fFieldSolver(nullptr),
    fDirectory(KEMFileInterface::GetInstance()->ActiveDirectory()),
    fHashMaskedBits(20),
    fHashThreshold(1.e-14)
{
    fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
    fFile = fFile.substr(fFile.find_last_of('/') + 1, std::string::npos);
}

KMagnetostaticBoundaryField::~KMagnetostaticBoundaryField() = default;

void KMagnetostaticBoundaryField::SetChargeDensitySolver(const std::shared_ptr<KMagneticChargeDensitySolver>& solver)
{
    fChargeDensitySolver = solver;
}

std::shared_ptr<KMagneticChargeDensitySolver> KMagnetostaticBoundaryField::GetChargeDensitySolver()
{
    return fChargeDensitySolver;
}

void KMagnetostaticBoundaryField::SetFieldSolver(const std::shared_ptr<KMagneticFFTMFieldSolver>& solver)
{
    fFieldSolver = solver;
}

std::shared_ptr<KMagneticFFTMFieldSolver> KMagnetostaticBoundaryField::GetFieldSolver()
{
    return fFieldSolver;
}

void KMagnetostaticBoundaryField::SetContainer(const std::shared_ptr<KSurfaceContainer>& container)
{
    fContainer = container;
}

std::shared_ptr<KSurfaceContainer> KMagnetostaticBoundaryField::GetContainer() const
{
    return fContainer;
}

KFieldVector KMagnetostaticBoundaryField::MagneticFieldCore(const KPosition& P) const
{
    return fFieldSolver->MagneticField(P);
}

KFieldVector KMagnetostaticBoundaryField::MagneticPotentialCore(const KPosition& P) const
{
    return MagneticPotentialCore(P);
}

KGradient KMagnetostaticBoundaryField::MagneticGradientCore(const KPosition& P) const
{
    return MagneticGradientCore(P);
}




void KMagnetostaticBoundaryField::InitializeCore()
{

    CheckSolverExistance();
//    fChargeDensitySolver->SetHashProperties(fHashMaskedBits, fHashThreshold);

    KEMFileInterface::GetInstance()->ActiveDirectory(fDirectory);
    KEMFileInterface::GetInstance()->ActiveFile(KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFile);

    VisitorPreprocessing();

    fChargeDensitySolver->Initialize(*fContainer);

    VisitorInBetweenProcessing();

    fFieldSolver->Initialize(*fContainer);

    VisitorPostprocessing();
}

void KMagnetostaticBoundaryField::DeinitializeCore()
{
    fFieldSolver->Deinitialize();
}

void KMagnetostaticBoundaryField::AddVisitor(const std::shared_ptr<Visitor>& visitor)
{
    fVisitors.push_back(visitor);
}

vector<std::shared_ptr<KMagnetostaticBoundaryField::Visitor>> KMagnetostaticBoundaryField::GetVisitors()
{
    return fVisitors;
}

void KMagnetostaticBoundaryField::SetDirectory(const string& aDirectory)
{
    fDirectory = aDirectory;
    return;
}
void KMagnetostaticBoundaryField::SetFile(const string& aFile)
{
    fFile = aFile;
    return;
}

void KMagnetostaticBoundaryField::SetHashMaskedBits(const unsigned int& aMaskedBits)
{
    fHashMaskedBits = aMaskedBits;
    return;
}
void KMagnetostaticBoundaryField::SetHashThreshold(const double& aThreshold)
{
    fHashThreshold = aThreshold;
    return;
}

void KMagnetostaticBoundaryField::VisitorPreprocessing()
{
    for (auto& visitor : fVisitors)
        if (visitor->Preprocessing())
            visitor->PreVisit(*this);
}

void KMagnetostaticBoundaryField::VisitorInBetweenProcessing()
{
    for (auto& visitor : fVisitors)
        if (visitor->InBetweenProcessing())
            visitor->InBetweenVisit(*this);
}

void KMagnetostaticBoundaryField::VisitorPostprocessing()
{
    for (auto& visitor : fVisitors)
        if (visitor->Postprocessing())
            visitor->PostVisit(*this);
}

void KMagnetostaticBoundaryField::CheckSolverExistance()
{
    if (!fChargeDensitySolver)
        throw KEMSimpleException("Initializing aborted: no charge density solver!");

    if (!fFieldSolver)
        throw KEMSimpleException("Initializing aborted: no field solver!");
}
//**********
//visitor
//**********

KMagnetostaticBoundaryField::Visitor::Visitor() :
    fPreprocessing(false),
    fInBetweenProcessing(false),
    fPostprocessing(false)
{}

void KMagnetostaticBoundaryField::Visitor::Preprocessing(bool choice)
{
    fPreprocessing = choice;
}

void KMagnetostaticBoundaryField::Visitor::InBetweenProcessing(bool choice)
{
    fInBetweenProcessing = choice;
}

void KMagnetostaticBoundaryField::Visitor::Postprocessing(bool choice)
{
    fPostprocessing = choice;
}

bool KMagnetostaticBoundaryField::Visitor::Preprocessing() const
{
    return fPreprocessing;
}

bool KMagnetostaticBoundaryField::Visitor::InBetweenProcessing() const
{
    return fInBetweenProcessing;
}

bool KMagnetostaticBoundaryField::Visitor::Postprocessing() const
{
    return fPostprocessing;
}

}  // namespace KEMField
