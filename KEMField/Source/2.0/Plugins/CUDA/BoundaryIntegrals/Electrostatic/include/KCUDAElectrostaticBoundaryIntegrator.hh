#ifndef KCUDAELECTROSTATICBOUNDARYINTEGRATOR_DEF
#define KCUDAELECTROSTATICBOUNDARYINTEGRATOR_DEF

#include "kEMField_ElectrostaticBoundaryIntegrals_kernel.cuh"
#include "KCUDABoundaryIntegrator.hh"
#include "KSurfaceVisitors.hh"
#include "KElectrostaticBoundaryIntegrator.hh"

namespace KEMField
{

  class KCUDAElectrostaticBoundaryIntegrator :
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

    using KSelectiveVisitor<KShapeVisitor, KTYPELIST_4(KTriangle,
						       KRectangle,
						       KLineSegment,
						       KConicSection)>::Visit;

    KCUDAElectrostaticBoundaryIntegrator(KCUDASurfaceContainer& c);
    ~KCUDAElectrostaticBoundaryIntegrator();

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
 
  private:
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
  };

  template <class SourceShape>
  double KCUDAElectrostaticBoundaryIntegrator::Potential(const SourceShape* source,const KPosition& aPosition) const
  {
    StreamSourceToDevice(source);
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

    int global(1);
    int local(1);

    if( KEMFIELD_OCCUPANCYAPI ) {
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, PotentialBIKernel, 0, 0);
        std::cout << "[PotentialBIKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[PotentialBIKernel] Set block size: " << local << std::endl;
    }

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
  KEMThreeVector KCUDAElectrostaticBoundaryIntegrator::ElectricField(const SourceShape* source, const KPosition& aPosition) const
  {
    StreamSourceToDevice(source);

    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

    int global(1);
    int local(1);

    if( KEMFIELD_OCCUPANCYAPI ) {
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ElectricFieldBIKernel, 0, 0);
        std::cout << "[ElectricFieldBIKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[ElectricFieldBIKernel] Set block size: " << local << std::endl;
    }

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
  void KCUDAElectrostaticBoundaryIntegrator::ComputeBoundaryIntegral(SourceShape& source)
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

#endif /* KCUDAELECTROSTATICBOUNDARYINTEGRATOR_DEF */
