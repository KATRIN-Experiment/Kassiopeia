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

void KKrylovPreconditionerGenerator::SetMatrixGenerator(const std::shared_ptr<MatrixGenerator>& matrixGen)
{
    fMatrixGenerator = matrixGen;
}

std::shared_ptr<const KKrylovPreconditionerGenerator::MatrixGenerator>
KKrylovPreconditionerGenerator::GetMatrixGenerator() const
{
    return fMatrixGenerator;
}

void KKrylovPreconditionerGenerator::SetPreconditionerGenerator(const std::shared_ptr<MatrixGenerator>& preconGen)
{
    fPreconditionerGenerator = preconGen;
}

std::shared_ptr<KSquareMatrix<KKrylovPreconditionerGenerator::ValueType>>
KKrylovPreconditionerGenerator::Build(const KSurfaceContainer& container) const
{
    if (container.empty()) {
        kem_cout(eError) << "ERROR: Krylov preconditioner got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
    }

    std::shared_ptr<KSquareMatrix<ValueType>> A = fMatrixGenerator->Build(container);
    std::shared_ptr<KSquareMatrix<ValueType>> P;
    if (fPreconditionerGenerator)
        P = fPreconditionerGenerator->Build(container);
    return KBuildKrylovPreconditioner<ValueType>(fKrylovConfig, A, P);
}

} /* namespace KEMField */
