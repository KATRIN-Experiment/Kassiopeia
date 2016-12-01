#ifndef KCUDAELECTROSTATICNUMERICBOUNDARYINTEGRATOR_DEF
#define KCUDAELECTROSTATICNUMERICBOUNDARYINTEGRATOR_DEF

#include "KElectrostaticBoundaryIntegrator.hh"
#include "kEMField_ElectrostaticNumericBoundaryIntegrals_kernel.cuh"

#include "KCUDABoundaryIntegrator.hh"
#include "KSurfaceVisitors.hh"

namespace KEMField
{
class ElectrostaticCUDA;

class KCUDAElectrostaticNumericBoundaryIntegrator :
		public KCUDABoundaryIntegrator<KElectrostaticBasis>,
		public KSelectiveVisitor<KShapeVisitor, KTYPELIST_4(KTriangle,
				KRectangle,
				KLineSegment,
				KConicSection)>
{
public:
	typedef KElectrostaticBasis Basis;
	typedef Basis::ValueType ValueType;
	typedef KBoundaryType<Basis,KDirichletBoundary> DirichletBoundary;
	typedef KBoundaryType<Basis,KNeumannBoundary> NeumannBoundary;

	// for selection of the correct KIntegratingFieldSolver template and possibly elsewhere
	typedef ElectrostaticCUDA Kind;
	typedef KElectrostaticBoundaryIntegrator IntegratorSingleThread;

	using KSelectiveVisitor<KShapeVisitor, KTYPELIST_4(KTriangle,
			KRectangle,
			KLineSegment,
			KConicSection)>::Visit;

	KCUDAElectrostaticNumericBoundaryIntegrator(KCUDASurfaceContainer& c);
	~KCUDAElectrostaticNumericBoundaryIntegrator();

	void Visit(KTriangle& t) { ComputeBoundaryIntegral(t); }
	void Visit(KRectangle& r) { ComputeBoundaryIntegral(r); }
	void Visit(KLineSegment& l) { ComputeBoundaryIntegral(l); }
	void Visit(KConicSection& c) { ComputeBoundaryIntegral(c); }

	ValueType  BoundaryIntegral(KSurfacePrimitive* source,
			KSurfacePrimitive* target,
			unsigned int);
	ValueType  BoundaryValue(KSurfacePrimitive* surface,unsigned int);
	ValueType& BasisValue(KSurfacePrimitive* surface,unsigned int);

	template <class SourceShape>
	double Potential(const SourceShape*, const KPosition&) const;
	template <class SourceShape>
	KEMThreeVector ElectricField(const SourceShape*, const KPosition&) const;
    template <class SourceShape>
    std::pair<KEMThreeVector,double> ElectricFieldAndPotential(const SourceShape*, const KPosition&) const;

	class BoundaryVisitor :
			public KSelectiveVisitor<KBoundaryVisitor,
			KTYPELIST_2(KDirichletBoundary,KNeumannBoundary)>
	{
	public:
		using KSelectiveVisitor<KBoundaryVisitor,KTYPELIST_2(KDirichletBoundary,KNeumannBoundary)>::Visit;

		BoundaryVisitor() {}
		virtual ~BoundaryVisitor() {}

		void Visit(KDirichletBoundary&);
		void Visit(KNeumannBoundary&);

		bool IsDirichlet() const { return fIsDirichlet; }
		ValueType Prefactor() const { return fPrefactor; }
		ValueType GetBoundaryValue() const { return fBoundaryValue; }

	protected:

		bool fIsDirichlet;
		ValueType fPrefactor;
		ValueType fBoundaryValue;
	};

	class BasisVisitor :
			public KSelectiveVisitor<KBasisVisitor,KTYPELIST_1(KElectrostaticBasis)>
	{
	public:
		using KSelectiveVisitor<KBasisVisitor,
				KTYPELIST_1(KElectrostaticBasis)>::Visit;

		BasisVisitor() : fBasisValue(NULL) {}
		virtual ~BasisVisitor() {}

		void Visit(KElectrostaticBasis&);

		ValueType& GetBasisValue() const { return *fBasisValue; }

	protected:

		ValueType* fBasisValue;
	};

	template <class SourceShape>
	void ComputeBoundaryIntegral(SourceShape& source);

	BoundaryVisitor fBoundaryVisitor;
	BasisVisitor fBasisVisitor;
	KSurfacePrimitive* fTarget;
	ValueType fValue;

  private:

	void ConstructCUDAKernels() const;
	void AssignDeviceMemory() const;

	mutable double *fDevicePhi;
	mutable CU_TYPE4 *fDeviceEField;
	mutable CU_TYPE4 *fDeviceEFieldAndPhi;
};

template <class SourceShape>
double KCUDAElectrostaticNumericBoundaryIntegrator::Potential(const SourceShape* source,const KPosition& aPosition) const
{
	StreamSourceToDevice(source);

	CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
	cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

	int global(1);
	int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
		int blockSize = 0.;   // The launch configurator returned block size
		int minGridSize = 0.; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, PotentialBIKernel, 0, 0);
		std::cout << "[PotentialBIKernel] Suggested block size: " << blockSize << std::endl;
		std::cout << "[PotentialBIKernel] Set block size: " << local << std::endl;
#endif

	PotentialBIKernel <<<global, local>>> (
			fDeviceP,
			fDeviceShapeInfo,
			fDeviceShapeData,
			fDevicePhi);

	CU_TYPE phi = 0.;
	cudaMemcpy( &phi, fDevicePhi, sizeof(CU_TYPE), cudaMemcpyDeviceToHost );

	return phi;
}

template <class SourceShape>
KEMThreeVector KCUDAElectrostaticNumericBoundaryIntegrator::ElectricField(const SourceShape* source, const KPosition& aPosition) const
{
	StreamSourceToDevice(source);

	CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

	cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

	int global(1);
	int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
		int blockSize = 0;   // The launch configurator returned block size
		int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ElectricFieldBIKernel, 0, 0);
		std::cout << "[ElectricFieldBIKernel] Suggested block size: " << blockSize << std::endl;
		std::cout << "[ElectricFieldBIKernel] Set block size: " << local << std::endl;
#endif

	ElectricFieldBIKernel <<<global, local>>> (
			fDeviceP,
			fDeviceShapeInfo,
			fDeviceShapeData,
			fDeviceEField );

	CU_TYPE4 eField;
	cudaMemcpy( &eField, fDeviceEField, sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

	return KEMThreeVector(eField.x,eField.y,eField.z);
}

template <class SourceShape>
std::pair<KEMThreeVector,double> KCUDAElectrostaticNumericBoundaryIntegrator::ElectricFieldAndPotential(const SourceShape* source, const KPosition& aPosition) const
{
	StreamSourceToDevice(source);

	CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

	cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

	int global(1);
	int local(1);

#ifdef KEMFIELD_OCCUPANCYAPI
		int blockSize = 0;   // The launch configurator returned block size
		int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
		cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ElectricFieldAndPotentialBIKernel, 0, 0);
		std::cout << "[ElectricFieldAndPotentialBIKernel] Suggested block size: " << blockSize << std::endl;
		std::cout << "[ElectricFieldAndPotentialBIKernel] Set block size: " << local << std::endl;
#endif

	ElectricFieldAndPotentialBIKernel <<<global, local>>> (
			fDeviceP,
			fDeviceShapeInfo,
			fDeviceShapeData,
			fDeviceEFieldAndPhi );

	CU_TYPE4 eFieldAndPhi;
	cudaMemcpy( &eFieldAndPhi, fDeviceEFieldAndPhi, sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

	return std::make_pair( KEMThreeVector(eFieldAndPhi.x,eFieldAndPhi.y,eFieldAndPhi.z), eFieldAndPhi.w);
}

template <class SourceShape>
void KCUDAElectrostaticNumericBoundaryIntegrator::ComputeBoundaryIntegral(SourceShape& source)
{
	if (fBoundaryVisitor.IsDirichlet()) {
		fValue = this->Potential(&source,fTarget->GetShape()->Centroid());
	}
	else
	{
		KEMThreeVector field = this->ElectricField(&source,fTarget->GetShape()->Centroid());
		fValue = field.Dot(fTarget->GetShape()->Normal());
		double dist = (source.Centroid()-fTarget->GetShape()->Centroid()).Magnitude();

		if (dist<1.e-12)
			fValue *= fBoundaryVisitor.Prefactor();
	}
}
}

#endif /* KCUDAELECTROSTATICNUMERICBOUNDARYINTEGRATOR_DEF */
