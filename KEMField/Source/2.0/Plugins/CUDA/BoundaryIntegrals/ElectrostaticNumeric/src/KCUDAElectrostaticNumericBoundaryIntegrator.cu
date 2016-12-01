#include <iostream>
#include <fstream>
#include <sstream>

#include "KCUDAElectrostaticNumericBoundaryIntegrator.hh"
#include "KSurfaceTypes.hh"
#include "KCUDASurfaceContainer.hh"


namespace KEMField
{

KCUDAElectrostaticNumericBoundaryIntegrator::KCUDAElectrostaticNumericBoundaryIntegrator(KCUDASurfaceContainer& c) :
		KCUDABoundaryIntegrator<KElectrostaticBasis>(c),
		fDevicePhi(NULL),
		fDeviceEField(NULL),
		fDeviceEFieldAndPhi(NULL)
{
	ConstructCUDAKernels();
	AssignDeviceMemory();
}

KCUDAElectrostaticNumericBoundaryIntegrator::~KCUDAElectrostaticNumericBoundaryIntegrator()
{
	if( fDevicePhi ) cudaFree(fDevicePhi);
	if( fDeviceEField ) cudaFree(fDeviceEField);
	if( fDeviceEFieldAndPhi ) cudaFree(fDeviceEFieldAndPhi);
}

void KCUDAElectrostaticNumericBoundaryIntegrator::BoundaryVisitor::Visit(KDirichletBoundary& boundary)
{
	fIsDirichlet = true;
	fPrefactor = 1.;
	fBoundaryValue = static_cast<DirichletBoundary&>(boundary).GetBoundaryValue();
}

void KCUDAElectrostaticNumericBoundaryIntegrator::BoundaryVisitor::Visit(KNeumannBoundary& boundary)
{
	fIsDirichlet = false;
	fPrefactor = ((1. + static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux())/(1. - static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()));
	fBoundaryValue = 0.;
}

void KCUDAElectrostaticNumericBoundaryIntegrator::BasisVisitor::Visit(KElectrostaticBasis& basis)
{
	fBasisValue = &(basis.GetSolution());
}

KElectrostaticBasis::ValueType KCUDAElectrostaticNumericBoundaryIntegrator::BoundaryIntegral(KSurfacePrimitive* source,KSurfacePrimitive* target,unsigned int)
{
	fTarget = target;
	target->Accept(fBoundaryVisitor);
	source->Accept(*this);
	return fValue;
}

KElectrostaticBasis::ValueType KCUDAElectrostaticNumericBoundaryIntegrator::BoundaryValue(KSurfacePrimitive* surface,unsigned int)
{
	surface->Accept(fBoundaryVisitor);
	return fBoundaryVisitor.GetBoundaryValue();
}

KElectrostaticBasis::ValueType& KCUDAElectrostaticNumericBoundaryIntegrator::BasisValue(KSurfacePrimitive* surface,unsigned int)
{
	surface->Accept(fBasisVisitor);
	return fBasisVisitor.GetBasisValue();
}

void KCUDAElectrostaticNumericBoundaryIntegrator::ConstructCUDAKernels() const
{
	KCUDABoundaryIntegrator<KElectrostaticBasis>::ConstructCUDAKernels();

	// Create memory buffers

	cudaMalloc( (void**) &fDeviceP, 3*sizeof(CU_TYPE) );
	cudaMalloc( (void**) &fDeviceShapeInfo, sizeof(short));
	// Hard-coded arbitrary maximum shape limit
	cudaMalloc( (void**) &fDeviceShapeData, 20*sizeof(CU_TYPE) );

	cudaMalloc( (void**) &fDevicePhi, sizeof(CU_TYPE) );
	cudaMalloc( (void**) &fDeviceEField, sizeof(CU_TYPE4) );
	cudaMalloc( (void**) &fDeviceEFieldAndPhi, sizeof(CU_TYPE4) );

	// copy weights and nodes to constant memory, get values from CPU integrator class
	// constant variables have been defined in corresponding device functions

    // n.b. CUDA memcopy needed also in executables in order
    // to guarantee that data will be copied onto GPU constant memory

    // 7-point triangle cubature

    cudaMemcpyToSymbol( cuTriCub7alpha, gTriCub7alpha, sizeof(CU_TYPE)*3 );
    cudaMemcpyToSymbol( cuTriCub7beta, gTriCub7beta, sizeof(CU_TYPE)*3 );
    cudaMemcpyToSymbol( cuTriCub7gamma, gTriCub7gamma, sizeof(CU_TYPE)*3 );
    cudaMemcpyToSymbol( cuTriCub7w, gTriCub7w, sizeof(CU_TYPE)*7 );

    // 12-point triangle cubature

    cudaMemcpyToSymbol( cuTriCub12alpha, gTriCub12alpha, sizeof(CU_TYPE)*4 );
    cudaMemcpyToSymbol( cuTriCub12beta, gTriCub12beta, sizeof(CU_TYPE)*4 );
    cudaMemcpyToSymbol( cuTriCub12gamma, gTriCub12gamma, sizeof(CU_TYPE)*4 );
    cudaMemcpyToSymbol( cuTriCub12w, gTriCub12w, sizeof(CU_TYPE)*12 );

    // 33-point triangle cubature

    cudaMemcpyToSymbol( cuTriCub33alpha, gTriCub33alpha, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuTriCub33beta, gTriCub33beta, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuTriCub33gamma, gTriCub33gamma, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuTriCub33w, gTriCub33w, sizeof(CU_TYPE)*33 );

    // rectangle cubature weights

    cudaMemcpyToSymbol( cuRectCub7w, gRectCub7w, sizeof(CU_TYPE)*7 );
    cudaMemcpyToSymbol( cuRectCub12w, gRectCub12w, sizeof(CU_TYPE)*12 );
    cudaMemcpyToSymbol( cuRectCub33w, gRectCub33w, sizeof(CU_TYPE)*33 );

    // quadrature weights and nodes for line segments

    cudaMemcpyToSymbol( cuLineQuadx4, gQuadx4, sizeof(CU_TYPE)*2 );
    cudaMemcpyToSymbol( cuLineQuadw4, gQuadw4, sizeof(CU_TYPE)*2 );
    cudaMemcpyToSymbol( cuLineQuadx16, gQuadx16, sizeof(CU_TYPE)*8 );
    cudaMemcpyToSymbol( cuLineQuadw16, gQuadw16, sizeof(CU_TYPE)*8 );
}

void KCUDAElectrostaticNumericBoundaryIntegrator::AssignDeviceMemory() const
{
	KCUDABoundaryIntegrator<KElectrostaticBasis>::AssignDeviceMemory();
}

}
