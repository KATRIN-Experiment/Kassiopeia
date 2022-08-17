/*
 * KKrylovPreconditioner.cc
 *
 *  Created on: 18 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovPreconditionerGenerator.hh"

#include "KImplicitKrylovPreconditioner.hh"

namespace KEMField
{

KKrylovPreconditionerGenerator::KKrylovPreconditionerGenerator()
{
    fKrylovConfig.SetDisplayName("Preconditioner: ");
}

KKrylovPreconditionerGenerator::~KKrylovPreconditionerGenerator() = default;

void KKrylovPreconditionerGenerator::SetMatrixGenerator(const KSmartPointer<MatrixGenerator>& matrixGen)
{
    fMatrixGenerator = matrixGen;
}

KSmartPointer<const KKrylovPreconditionerGenerator::MatrixGenerator>
KKrylovPreconditionerGenerator::GetMatrixGenerator() const
{
    return fMatrixGenerator;
}

void KKrylovPreconditionerGenerator::SetPreconditionerGenerator(const KSmartPointer<MatrixGenerator>& preconGen)
{
    fPreconditionerGenerator = preconGen;
}

KSmartPointer<KSquareMatrix<KKrylovPreconditionerGenerator::ValueType>>
KKrylovPreconditionerGenerator::Build(const KSurfaceContainer& container) const
{
    if (container.empty()) {
        kem_cout(eError) << "ERROR: Krylov preconditioner got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
    }

    KSmartPointer<KSquareMatrix<ValueType>> A = fMatrixGenerator->Build(container);
    KSmartPointer<KSquareMatrix<ValueType>> P;
    if (fPreconditionerGenerator.Is())
        P = fPreconditionerGenerator->Build(container);
    return KBuildKrylovPreconditioner<ValueType>(fKrylovConfig, A, P);
}

} /* namespace KEMField */
