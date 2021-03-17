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

KSmartPointer<KSquareMatrix<KFastMultipoleMatrixGenerator::ValueType>>
KFastMultipoleMatrixGenerator::Build(const KSurfaceContainer& container) const
{
    KSmartPointer<FastMultipoleEBI> fm_integrator(new FastMultipoleEBI(fDirectIntegrator, container));
    KSmartPointer<FastMultipoleMatrix> fmA = CreateMatrix(container, fm_integrator);
    return fmA;
}

KSmartPointer<FastMultipoleMatrix>
KFastMultipoleMatrixGenerator::CreateMatrix(const KSurfaceContainer& surfaceContainer,
                                            const KSmartPointer<FastMultipoleEBI>& fm_integrator) const
{
    fm_integrator->Initialize(fParameters);

    KSmartPointer<FastMultipoleSparseMatrix> sparseA(new FastMultipoleSparseMatrix(surfaceContainer, fm_integrator));

    KSmartPointer<FastMultipoleDenseMatrix> denseA(new FastMultipoleDenseMatrix(fm_integrator));

    return KSmartPointer<FastMultipoleMatrix>(new FastMultipoleMatrix(denseA, sparseA));
}

} /* namespace KEMField */
