/*
 * KFastMultipoleMatrixGenerator.cc
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#include "KFastMultipoleMatrixGenerator.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"

using namespace KEMField::KFMElectrostaticTypes;

namespace KEMField
{

KFastMultipoleMatrixGenerator::KFastMultipoleMatrixGenerator() : fDirectIntegrator{KEBIFactory::MakeDefaultForFFTM()}
{
    // TODO Auto-generated constructor stub
}

KFastMultipoleMatrixGenerator::~KFastMultipoleMatrixGenerator() = default;
//{
//    TODO Auto-generated destructor stub
//}

std::shared_ptr<KSquareMatrix<KFastMultipoleMatrixGenerator::ValueType>>
KFastMultipoleMatrixGenerator::Build(const KSurfaceContainer& container) const
{
    std::shared_ptr<FastMultipoleEBI> fm_integrator(new FastMultipoleEBI(fDirectIntegrator, container));
    auto fmA = CreateMatrix(container, fm_integrator);
    return fmA;
}

std::shared_ptr<FastMultipoleMatrix>
KFastMultipoleMatrixGenerator::CreateMatrix(const KSurfaceContainer& surfaceContainer,
                                            const std::shared_ptr<FastMultipoleEBI>& fm_integrator) const
{
    fm_integrator->Initialize(fParameters);

    std::shared_ptr<FastMultipoleSparseMatrix> sparseA(new FastMultipoleSparseMatrix(surfaceContainer, fm_integrator));

    std::shared_ptr<FastMultipoleDenseMatrix> denseA(new FastMultipoleDenseMatrix(fm_integrator));

    return std::make_shared<FastMultipoleMatrix>(denseA, sparseA);
}

} /* namespace KEMField */
