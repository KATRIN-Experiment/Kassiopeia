#include "KPETScInterface.hh"

#include "KMPIInterface.hh"

#include <iostream>

namespace KEMField
{
KPETScInterface* KPETScInterface::fPETScInterface = 0;

KPETScInterface::KPETScInterface() {}

KPETScInterface::~KPETScInterface() {}

/**
   * Interface to accessing KPETScInterface.
   */
KPETScInterface* KPETScInterface::GetInstance()
{
    if (fPETScInterface == 0)
        fPETScInterface = new KPETScInterface();
    return fPETScInterface;
}

PetscErrorCode KPETScInterface::Initialize(int* argc, char*** argv)
{
    static char help[] = "KEMField PETSc interface.\n\n";
    PetscErrorCode iErr = PetscInitialize(argc, argv, PETSC_NULL, help);
    KMPIInterface::GetInstance()->Initialize(argc, argv);
    return iErr;
}

PetscErrorCode KPETScInterface::Finalize()
{
    PetscErrorCode iErr = PetscFinalize();
    KMPIInterface::GetInstance()->Finalize();
    return iErr;
}

}  // namespace KEMField
