/*
 * KMagfieldCoilsFieldSolverBuilder.hh
 *
 *  Created on: 31 Jan 2023
 *      Author: Jan Behrens
 */

#ifndef KMagfieldCoilsFieldSolverBuilder_HH_
#define KMagfieldCoilsFieldSolverBuilder_HH_

#include "KComplexElement.hh"
#include "KMagfieldCoilsFieldSolver.hh"
namespace katrin
{

typedef KComplexElement<KEMField::KMagfieldCoilsFieldSolver> KMagfieldCoilsFieldSolverBuilder;

template<> inline bool KMagfieldCoilsFieldSolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetObjectName);
        return true;
    }
    if (aContainer->GetName() == "directory") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetDirName);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetCoilFileName);
        return true;
    }
    if (aContainer->GetName() == "replace_file") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetReplaceFile);
        return true;
    }
    if (aContainer->GetName() == "force_elliptic") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetForceElliptic);
        return true;
    }
    if (aContainer->GetName() == "n_elliptic") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetNElliptic);
        return true;
    }
    if (aContainer->GetName() == "n_max") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetNMax);
        return true;
    }
    if (aContainer->GetName() == "eps_tol") {
        aContainer->CopyTo(fObject, &KEMField::KMagfieldCoilsFieldSolver::SetEpsTol);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KMagfieldCoilsFieldSolverBuilder_HH_ */
