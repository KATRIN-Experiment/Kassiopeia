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

KKrylovPreconditionerGenerator::~KKrylovPreconditionerGenerator() {}

void KKrylovPreconditionerGenerator::SetMatrixGenerator(KSmartPointer<MatrixGenerator> matrixGen)
{
    fMatrixGenerator = matrixGen;
}

KSmartPointer<const KKrylovPreconditionerGenerator::MatrixGenerator>
KKrylovPreconditionerGenerator::GetMatrixGenerator() const
{
    return fMatrixGenerator;
}

void KKrylovPreconditionerGenerator::SetPreconditionerGenerator(KSmartPointer<MatrixGenerator> preconGen)
{
    fPreconditionerGenerator = preconGen;
}

KSmartPointer<KSquareMatrix<KKrylovPreconditionerGenerator::ValueType>>
KKrylovPreconditionerGenerator::Build(const KSurfaceContainer& container) const
{
    KSmartPointer<KSquareMatrix<ValueType>> A = fMatrixGenerator->Build(container);
    KSmartPointer<KSquareMatrix<ValueType>> P;
    if (fPreconditionerGenerator.Is())
        P = fPreconditionerGenerator->Build(container);
    return KBuildKrylovPreconditioner<ValueType>(fKrylovConfig, A, P);
}

} /* namespace KEMField */
