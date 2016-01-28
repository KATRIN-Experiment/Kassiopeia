#include <iostream>
#include <fstream>
#include <sstream>

#include "KCUDAElectrostaticBoundaryIntegrator.hh"
#include "KSurfaceTypes.hh"
#include "KCUDASurfaceContainer.hh"

namespace KEMField
{
  KCUDAElectrostaticBoundaryIntegrator::KCUDAElectrostaticBoundaryIntegrator(KCUDASurfaceContainer& c) :
    KCUDABoundaryIntegrator<KElectrostaticBasis>(c),
    fDevicePhi(NULL),
    fDeviceEField(NULL)
  {
    ConstructCUDAKernels();
    AssignDeviceMemory();
  }

  KCUDAElectrostaticBoundaryIntegrator::~KCUDAElectrostaticBoundaryIntegrator()
  {
    if( fDevicePhi ) cudaFree(fDevicePhi);
    if( fDeviceEField ) cudaFree(fDeviceEField);
  }

  void KCUDAElectrostaticBoundaryIntegrator::BoundaryVisitor::Visit(KDirichletBoundary& boundary)
  {
    fIsDirichlet = true;
    fPrefactor = 1.;
    fBoundaryValue = static_cast<DirichletBoundary&>(boundary).GetBoundaryValue();
  }

  void KCUDAElectrostaticBoundaryIntegrator::BoundaryVisitor::Visit(KNeumannBoundary& boundary)
  {
    fIsDirichlet = false;
    fPrefactor = ((1. + static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux())/(1. - static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()));
    fBoundaryValue = 0.;
  }

  void KCUDAElectrostaticBoundaryIntegrator::BasisVisitor::Visit(KElectrostaticBasis& basis)
  {
    fBasisValue = &(basis.GetSolution());
  }

  KElectrostaticBasis::ValueType KCUDAElectrostaticBoundaryIntegrator::BoundaryIntegral(KSurfacePrimitive* source,KSurfacePrimitive* target,unsigned int)
  {
    fTarget = target;
    target->Accept(fBoundaryVisitor);
    source->Accept(*this);
    return fValue;
  }

  KElectrostaticBasis::ValueType KCUDAElectrostaticBoundaryIntegrator::BoundaryValue(KSurfacePrimitive* surface,unsigned int)
  {
    surface->Accept(fBoundaryVisitor);
    return fBoundaryVisitor.GetBoundaryValue();
  }

  KElectrostaticBasis::ValueType& KCUDAElectrostaticBoundaryIntegrator::BasisValue(KSurfacePrimitive* surface,unsigned int)
  {
    surface->Accept(fBasisVisitor);
    return fBasisVisitor.GetBasisValue();
  }

  void KCUDAElectrostaticBoundaryIntegrator::ConstructCUDAKernels() const
  {
    KCUDABoundaryIntegrator<KElectrostaticBasis>::ConstructCUDAKernels();

    // Create memory buffers
    cudaMalloc( (void**) &fDeviceP, 3*sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDeviceShapeInfo, sizeof(short ));
    // Hard-coded arbitrary maximum shape limit
    cudaMalloc( (void**) &fDeviceShapeData, 20*sizeof(CU_TYPE) );

    cudaMalloc( (void**) &fDeviceShapeData, 20 * sizeof(CU_TYPE));
    cudaMalloc( (void**) &fDevicePhi, sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDeviceEField, sizeof(CU_TYPE4) );
  }

  void KCUDAElectrostaticBoundaryIntegrator::AssignDeviceMemory() const
  {
    KCUDABoundaryIntegrator<KElectrostaticBasis>::AssignDeviceMemory();


  }
}
