#ifndef KCUDAELECTROSTATICINTEGRATINGFIELDSOLVER_DEF
#define KCUDAELECTROSTATICINTEGRATINGFIELDSOLVER_DEF

#include "KCUDAAction.hh"
#include "KCUDASurfaceContainer.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KIntegratingFieldSolverTemplate.hh"

#include "kEMField_ElectrostaticIntegratingFieldSolver_kernel.cuh"

#include <limits.h>
#include <fstream>
#include <sstream>

#define MAX_SUBSET_SIZE 10000
#define MIN_SUBSET_SIZE 16

namespace KEMField
{
class ElectrostaticCUDA;

template <class Integrator>
class KIntegratingFieldSolver<Integrator,ElectrostaticCUDA> :
public KCUDAAction
{
public:
    typedef typename Integrator::Basis Basis;

    KIntegratingFieldSolver(KCUDASurfaceContainer& container,
            Integrator& integrator);

    //use this constructor when sub-set solving is to be used
    KIntegratingFieldSolver(KCUDASurfaceContainer& container,
            Integrator& integrator,
            unsigned int max_subset_size,
            unsigned int min_subset_size = 16);

    virtual ~KIntegratingFieldSolver();

    void ConstructCUDAKernels() const;
    void AssignDeviceMemory() const;

    double Potential(const KPosition& aPosition) const;
    KEMThreeVector ElectricField(const KPosition& aPosition) const;
    std::pair<KEMThreeVector,double> ElectricFieldAndPotential(const KPosition& aPosition) const;

    ////////////////////////////////////////////////////////////////////////////

    //sub-set potential/field calls
    double Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    KEMThreeVector ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    std::pair<KEMThreeVector,double> ElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;

    //these methods allow us to dispatch a calculation to the GPU and retrieve the values later
    //this is useful so that we can do other work while waiting for the results
    void DispatchPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    void DispatchElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    void DispatchElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const;
    double RetrievePotential() const;
    KEMThreeVector RetrieveElectricField() const;
    std::pair<KEMThreeVector,double> RetrieveElectricFieldAndPotential() const;

    ////////////////////////////////////////////////////////////////////////////

private:

    void GetReorderedSubsetIndices(const unsigned int* sorted_container_ids, unsigned int* normal_container_ids, unsigned int size) const;

    KCUDASurfaceContainer& fContainer;

    typedef typename Integrator::IntegratorSingleThread FallbackIntegrator;
    FallbackIntegrator fStandardIntegrator;
    KIntegratingFieldSolver<FallbackIntegrator> fStandardSolver;

    mutable CU_TYPE *fDeviceP;
    mutable CU_TYPE *fDevicePotential;
    mutable CU_TYPE4 *fDeviceElectricField;
    mutable CU_TYPE4 *fDeviceElectricFieldAndPotential;

    mutable CU_TYPE* fHostPotential;
    mutable CU_TYPE4* fHostElectricField;
    mutable CU_TYPE4* fHostElectricFieldAndPotential;

    mutable unsigned int fNGlobal;
    mutable unsigned int fNLocal;
    mutable unsigned int fNWorkgroups;

    ////////////////////////////////////////////////////////////////////////////

    //sub-set kernel and buffers
    unsigned int fMaxSubsetSize;
    unsigned int fMinSubsetSize;
    mutable unsigned int* fDeviceElementIdentities;
    mutable unsigned int* fSubsetIdentities;
    mutable unsigned int* fReorderedSubsetIdentities;

    //for dispatched and later collected subset potential/field
    mutable unsigned int fCachedNGlobal;
    mutable unsigned int fCachedNDummy;
    mutable unsigned int fCachedNWorkgroups;
    mutable double fCachedSubsetPotential;
    mutable KEMThreeVector fCachedSubsetField;
    mutable std::pair<KEMThreeVector,double> fCachedSubsetFieldAndPotential;

    mutable unsigned int fCachedSubsetSize;
    mutable const unsigned int* fCachedSurfaceIndexSet;
    mutable KPosition fCachedPosition;

    mutable bool fCallDevice;

};

template <class Integrator>
KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::
KIntegratingFieldSolver(KCUDASurfaceContainer& container,
        Integrator& integrator) :
        KCUDAAction(container),
        fContainer(container),
        fStandardSolver(container.GetSurfaceContainer(),fStandardIntegrator),
        fHostPotential(NULL),
        fHostElectricField(NULL),
        fHostElectricFieldAndPotential(NULL),
        fMaxSubsetSize(MAX_SUBSET_SIZE),
        fMinSubsetSize(MIN_SUBSET_SIZE),
        fDeviceElementIdentities(NULL)
        {
    fSubsetIdentities = new unsigned int[MAX_SUBSET_SIZE];
    fReorderedSubsetIdentities = new unsigned int[MAX_SUBSET_SIZE];
        }

template <class Integrator>
KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::
KIntegratingFieldSolver(KCUDASurfaceContainer& container,
        Integrator& integrator,
        unsigned int max_subset_size,
        unsigned int min_subset_size) :
        KCUDAAction(container),
        fContainer(container),
        fStandardSolver(container.GetSurfaceContainer(),fStandardIntegrator),
        fHostPotential(NULL),
        fHostElectricField(NULL),
        fHostElectricFieldAndPotential(NULL),
        fMaxSubsetSize(max_subset_size),
        fMinSubsetSize(min_subset_size),
        fDeviceElementIdentities(NULL)
        {
    if( fMaxSubsetSize == 0 ) {fMaxSubsetSize = MAX_SUBSET_SIZE;};
    fSubsetIdentities = new unsigned int[fMaxSubsetSize];
    fReorderedSubsetIdentities = new unsigned int[fMaxSubsetSize];
        }


template <class Integrator>
KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::
~KIntegratingFieldSolver()
{
    if( fHostPotential ) delete fHostPotential;
    if( fHostElectricField ) delete fHostElectricField;
    if( fHostElectricFieldAndPotential ) delete fHostElectricFieldAndPotential;
    if( fDeviceElementIdentities ) cudaFree(fDeviceElementIdentities);
    delete[] fSubsetIdentities;
    delete[] fReorderedSubsetIdentities;
}

template <class Integrator>
void KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::ConstructCUDAKernels() const
{
    fNLocal = 384;

    // make sure that the available local memory on the device is sufficient for the thread block size
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, KEMFIELD_DEFAULT_GPU_ID);
    long localMemory = devProp.sharedMemPerBlock;

    if (fNLocal * sizeof(CU_TYPE4) > localMemory)
        fNLocal = localMemory / sizeof(CU_TYPE4);

    //ensure that fNLocal is a power of two (necessary for parallel reduction)
    unsigned int proper_size = 1;
    do
    {
        proper_size *= 2;
    }
    while( 2*proper_size <= fNLocal );
    fNLocal = proper_size;

    fContainer.SetMinimumWorkgroupSizeForKernels(fNLocal);
}

template <class Integrator>
void KIntegratingFieldSolver<Integrator, ElectrostaticCUDA>::AssignDeviceMemory() const
{
    fNGlobal = fContainer.GetNBufferedElements();
    fNLocal = fContainer.GetMinimumWorkgroupSizeForKernels();
    fNWorkgroups = fContainer.GetNBufferedElements()/fNLocal;

    cudaMalloc( (void**) &fDeviceP, 3*sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDevicePotential, fNWorkgroups*sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDeviceElectricField, fNWorkgroups*sizeof(CU_TYPE4) );
    cudaMalloc( (void**) &fDeviceElectricFieldAndPotential, fNWorkgroups*sizeof(CU_TYPE4) );

    fHostPotential = new CU_TYPE[fNWorkgroups];
    fHostElectricField = new CU_TYPE4[fNWorkgroups];
    fHostElectricFieldAndPotential = new CU_TYPE4[fNWorkgroups];

    cudaMemcpy( fDevicePotential, fHostPotential, fNWorkgroups*sizeof(CU_TYPE), cudaMemcpyHostToDevice );
    cudaMemcpy( fDeviceElectricField, fHostElectricField, fNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyHostToDevice );
    cudaMemcpy( fDeviceElectricFieldAndPotential, fHostElectricFieldAndPotential, fNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyHostToDevice );

    //create the element id buffer
    cudaMalloc( (void**) &fDeviceElementIdentities, fMaxSubsetSize*sizeof(unsigned int) );
}

template <class Integrator>
double KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::Potential(const KPosition& aPosition) const
{
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

#ifdef KEMFIELD_OCCUPANCYAPI
    int blockSize = 0;   // The launch configurator returned block size
    int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, PotentialKernel, 0, 0);
    std::cout << "[PotentialKernel] Suggested block size: " << blockSize << std::endl;
    std::cout << "[PotentialKernel] Set block size: " << fNLocal << std::endl;
#endif

    PotentialKernel <<<fNWorkgroups, fNLocal, fNLocal*sizeof(CU_TYPE)>>> (
            fDeviceP,
            fContainer.GetShapeInfo(),
            fContainer.GetShapeData(),
            fContainer.GetBasisData(),
            fDevicePotential );

    CU_TYPE potential = 0.;

    cudaMemcpy( fHostPotential, fDevicePotential, fNWorkgroups * sizeof(CU_TYPE), cudaMemcpyDeviceToHost );

    for( unsigned int i=0; i<fNWorkgroups; i++ )
        potential += fHostPotential[i];

    if( potential != potential )
        return fStandardSolver.Potential(aPosition);

    return potential;
}

template <class Integrator>
KEMThreeVector KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::ElectricField(const KPosition& aPosition) const
{
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

#ifdef KEMFIELD_OCCUPANCYAPI
    int blockSize = 0;   // The launch configurator returned block size
    int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ElectricFieldKernel, 0, 0);
    std::cout << "[ElectricFieldKernel] Suggested block size: " << blockSize << std::endl;
    std::cout << "[ElectricFieldKernel] Set block size: " << fNLocal << std::endl;
#endif

    ElectricFieldKernel <<<fNWorkgroups, fNLocal, fNLocal*sizeof(CU_TYPE4)>>> (
            fDeviceP,
            fContainer.GetShapeInfo(),
            fContainer.GetShapeData(),
            fContainer.GetBasisData(),
            fDeviceElectricField );

    KEMThreeVector eField(0.,0.,0.);

    cudaMemcpy( fHostElectricField, fDeviceElectricField, fNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

    for( unsigned int i=0; i<fNWorkgroups; i++ ) {
        eField[0] += fHostElectricField[i].x;
        eField[1] += fHostElectricField[i].y;
        eField[2] += fHostElectricField[i].z;
    }

    if( eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2] )
        return fStandardSolver.ElectricField(aPosition);

    return eField;
}

template <class Integrator>
std::pair<KEMThreeVector,double> KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::ElectricFieldAndPotential(const KPosition& aPosition) const
{
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

#ifdef KEMFIELD_OCCUPANCYAPI
    int blockSize = 0;   // The launch configurator returned block size
    int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ElectricFieldAndPotentialKernel, 0, 0);
    std::cout << "[ElectricFieldAndPotentialKernel] Suggested block size: " << blockSize << std::endl;
    std::cout << "[ElectricFieldAndPotentialKernel] Set block size: " << fNLocal << std::endl;
#endif

    ElectricFieldAndPotentialKernel <<<fNWorkgroups, fNLocal, fNLocal*sizeof(CU_TYPE4)>>> (
            fDeviceP,
            fContainer.GetShapeInfo(),
            fContainer.GetShapeData(),
            fContainer.GetBasisData(),
            fDeviceElectricFieldAndPotential );

    KEMThreeVector eField(0.,0.,0.);
    CU_TYPE potential = 0.;

    cudaMemcpy( fHostElectricFieldAndPotential, fDeviceElectricFieldAndPotential, fNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

    for( unsigned int i=0; i<fNWorkgroups; i++ ) {
        eField[0] += fHostElectricFieldAndPotential[i].x;
        eField[1] += fHostElectricFieldAndPotential[i].y;
        eField[2] += fHostElectricFieldAndPotential[i].z;
        potential += fHostElectricFieldAndPotential[i].w;
    }

    if (eField[0] != eField[0] ||
    eField[1] != eField[1] ||
    eField[2] != eField[2] ||
    potential != potential )
      return fStandardSolver.ElectricFieldAndPotential(aPosition);


    return std::make_pair(eField,potential);
}

template <class Integrator>
double KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    // evaluation point
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

    if( SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize ){
        // write point
        cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

        // write the element ids
        cudaMemcpy( fDeviceElementIdentities, SurfaceIndexSet, SetSize*sizeof(unsigned int), cudaMemcpyHostToDevice );

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global%fNLocal);
        if( nDummy == fNLocal ){nDummy = 0;};
        n_global += nDummy;

        int global(n_global);
        int local(fNLocal);

        unsigned int n_workgroups = n_global/fNLocal;

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetPotentialKernel, 0, 0);
        std::cout << "[SubsetPotentialKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[SubsetPotentialKernel] Set block size: " << local << std::endl;
#endif

        //queue kernel (SetSize = number of elements)
        SubsetPotentialKernel <<<n_workgroups,local,fNLocal*sizeof(CU_TYPE)>>> (
                fDeviceP,
                fContainer.GetShapeInfo(),
                fContainer.GetShapeData(),
                fContainer.GetBasisData(),
                fDevicePotential,
                SetSize,
                fDeviceElementIdentities );

        CU_TYPE potential = 0.;

        cudaMemcpy( fHostPotential, fDevicePotential, n_workgroups*sizeof(CU_TYPE), cudaMemcpyDeviceToHost );

        for( unsigned int i=0; i<n_workgroups; i++ ){
            potential += fHostPotential[i];
        }

        if( potential != potential ) {
            GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
            return fStandardSolver.Potential(fSubsetIdentities, SetSize, aPosition);
        }

        return potential;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        return fStandardSolver.Potential(fSubsetIdentities, SetSize, aPosition);
    }
}

template <class Integrator>
KEMThreeVector KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    //evaluation point
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

    if( SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize ) {
        //write point
        cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

        //write the element ids
        cudaMemcpy( fDeviceElementIdentities, SurfaceIndexSet, SetSize*sizeof(unsigned int), cudaMemcpyHostToDevice );

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global%fNLocal);
        if(nDummy == fNLocal){nDummy = 0;};
        n_global += nDummy;

        int global(n_global);
        int local(fNLocal);

        unsigned int n_workgroups = n_global/fNLocal;

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetElectricFieldKernel, 0, 0);
        std::cout << "[SubsetElectricFieldKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[SubsetElectricFieldKernel] Set block size: " << local << std::endl;
#endif

        //queue kernel
        SubsetElectricFieldKernel <<<n_workgroups,local,fNLocal*sizeof(CU_TYPE4)>>> (
                fDeviceP,
                fContainer.GetShapeInfo(),
                fContainer.GetShapeData(),
                fContainer.GetBasisData(),
                fDeviceElectricField,
                SetSize,
                fDeviceElementIdentities );


        KEMThreeVector eField(0.,0.,0.);

        // events not needed, since cudaMemcpy is a synchronous fct. which blocks the CPU until copy is complete
        cudaMemcpy( fHostElectricField, fDeviceElectricField, n_workgroups*sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

        for( unsigned int i=0; i<n_workgroups; i++ ) {
            eField[0] += fHostElectricField[i].x;
            eField[1] += fHostElectricField[i].y;
            eField[2] += fHostElectricField[i].z;
        }

        if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2]) {
            GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
            return fStandardSolver.ElectricField(fSubsetIdentities, SetSize, aPosition);
        }

        return eField;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        return fStandardSolver.ElectricField(fSubsetIdentities, SetSize, aPosition);
    }
}

template <class Integrator>
std::pair<KEMThreeVector,double> KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::ElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    //evaluation point
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

    if( SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize ) {
        //write point
        cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

        //write the element ids
        cudaMemcpy( fDeviceElementIdentities, SurfaceIndexSet, SetSize*sizeof(unsigned int), cudaMemcpyHostToDevice );

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global%fNLocal);
        if(nDummy == fNLocal){nDummy = 0;};
        n_global += nDummy;

        int global(n_global);
        int local(fNLocal);

        unsigned int n_workgroups = n_global/fNLocal;

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize = 0;   // The launch configurator returned block size
        int minGridSize = 0; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetElectricFieldAndPotentialKernel, 0, 0);
        std::cout << "[SubsetElectricFieldAndPotentialKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[SubsetElectricFieldAndPotentialKernel] Set block size: " << local << std::endl;
#endif

        //queue kernel
        SubsetElectricFieldAndPotentialKernel <<<n_workgroups,local,fNLocal*sizeof(CU_TYPE4)>>> (
                fDeviceP,
                fContainer.GetShapeInfo(),
                fContainer.GetShapeData(),
                fContainer.GetBasisData(),
                fDeviceElectricFieldAndPotential,
                SetSize,
                fDeviceElementIdentities );


        KEMThreeVector eField(0.,0.,0.);
        CU_TYPE potential = 0.;

        // events not needed, since cudaMemcpy is a synchronous fct. which blocks the CPU until copy is complete
        cudaMemcpy( fHostElectricFieldAndPotential, fDeviceElectricFieldAndPotential, n_workgroups*sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

        for( unsigned int i=0; i<n_workgroups; i++ ) {
            eField[0] += fHostElectricFieldAndPotential[i].x;
            eField[1] += fHostElectricFieldAndPotential[i].y;
            eField[2] += fHostElectricFieldAndPotential[i].z;
            potential += fHostElectricFieldAndPotential[i].w;
        }

        if (eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2]) {
            GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
            return fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, SetSize, aPosition);
        }

        return std::make_pair(eField,potential);
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        return fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, SetSize, aPosition);
    }
}

template <class Integrator>
void KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::GetReorderedSubsetIndices(const unsigned int* sorted_container_ids, unsigned int* normal_container_ids, unsigned int size) const
{
    for( unsigned int i=0; i<size; i++ ) {
        normal_container_ids[i] = fContainer.GetNormalIndexFromSortedIndex(sorted_container_ids[i]);
    }
}

template <class Integrator>
void KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::DispatchPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    fCachedSurfaceIndexSet = SurfaceIndexSet;
    fCachedSubsetSize = SetSize;
    fCallDevice = false;
    fCachedPosition = aPosition;

    if( SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize ) {
        //evaluation point
        CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

        //write point
        cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

        //write the element ids
        cudaMemcpy( fDeviceElementIdentities, SurfaceIndexSet, SetSize*sizeof(unsigned int), cudaMemcpyHostToDevice );

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global%fNLocal);
        if(nDummy == fNLocal){nDummy = 0;};
        n_global += nDummy;

        int global(n_global);
        int local(fNLocal);

        unsigned int n_workgroups = n_global/fNLocal;

        fCachedNGlobal = n_global;
        fCachedNDummy = nDummy;
        fCachedNWorkgroups = n_workgroups;

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetPotentialKernel, 0, 0);
        std::cout << "[SubsetPotentialKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[SubsetPotentialKernel] Set block size: " << local << std::endl;
#endif

        //queue kernel (SetSize = number of elements)
        SubsetPotentialKernel <<<n_workgroups,local,fNLocal*sizeof(CU_TYPE)>>> (
                fDeviceP,
                fContainer.GetShapeInfo(),
                fContainer.GetShapeData(),
                fContainer.GetBasisData(),
                fDevicePotential,
                SetSize,
                fDeviceElementIdentities );

        fCallDevice = true;
        return;
    }
    else
    {
        fCallDevice = false;
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        fCachedSubsetPotential = fStandardSolver.Potential(fSubsetIdentities, SetSize, aPosition);
    }
}

template <class Integrator>
void KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::DispatchElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    fCachedSurfaceIndexSet = SurfaceIndexSet;
    fCachedSubsetSize = SetSize;
    fCallDevice = false;
    fCachedPosition = aPosition;

    if( SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize ) {
        CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

        //write point
        cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

        //write the element ids
        cudaMemcpy( fDeviceElementIdentities, SurfaceIndexSet, SetSize*sizeof(unsigned int), cudaMemcpyHostToDevice );

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global%fNLocal);
        if(nDummy == fNLocal){nDummy = 0;};
        n_global += nDummy;

        int global(n_global);
        int local(fNLocal);

        unsigned int n_workgroups = n_global/fNLocal;

        fCachedNGlobal = n_global;
        fCachedNDummy = nDummy;
        fCachedNWorkgroups = n_workgroups;

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetElectricFieldKernel, 0, 0);
        std::cout << "[SubsetElectricFieldKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[SubsetElectricFieldKernel] Set block size: " << local << std::endl;
#endif

        SubsetElectricFieldKernel <<<n_workgroups,local,fNLocal*sizeof(CU_TYPE4)>>> (
                fDeviceP,
                fContainer.GetShapeInfo(),
                fContainer.GetShapeData(),
                fContainer.GetBasisData(),
                fDeviceElectricField,
                SetSize,
                fDeviceElementIdentities );

        fCallDevice = true;
        return;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        fCachedSubsetField = fStandardSolver.ElectricField(fSubsetIdentities, SetSize, aPosition);
        fCallDevice = false;
    }

}

template <class Integrator>
void KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::DispatchElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
{
    fCachedSurfaceIndexSet = SurfaceIndexSet;
    fCachedSubsetSize = SetSize;
    fCallDevice = false;
    fCachedPosition = aPosition;

    if( SetSize > fMinSubsetSize && SetSize <= fMaxSubsetSize ) {
        CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};

        //write point
        cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

        //write the element ids
        cudaMemcpy( fDeviceElementIdentities, SurfaceIndexSet, SetSize*sizeof(unsigned int), cudaMemcpyHostToDevice );

        //set the global range and nWorkgroups
        //pad out n-global to be a multiple of the n-local
        unsigned int n_global = SetSize;
        unsigned int nDummy = fNLocal - (n_global%fNLocal);
        if(nDummy == fNLocal){nDummy = 0;};
        n_global += nDummy;

        int global(n_global);
        int local(fNLocal);

        unsigned int n_workgroups = n_global/fNLocal;

        fCachedNGlobal = n_global;
        fCachedNDummy = nDummy;
        fCachedNWorkgroups = n_workgroups;

#ifdef KEMFIELD_OCCUPANCYAPI
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetElectricFieldAndPotentialKernel, 0, 0);
        std::cout << "[SubsetElectricFieldAndPotentialKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[SubsetElectricFieldAndPotentialKernel] Set block size: " << local << std::endl;
#endif

        SubsetElectricFieldAndPotentialKernel <<<n_workgroups,local,fNLocal*sizeof(CU_TYPE4)>>> (
                fDeviceP,
                fContainer.GetShapeInfo(),
                fContainer.GetShapeData(),
                fContainer.GetBasisData(),
                fDeviceElectricFieldAndPotential,
                SetSize,
                fDeviceElementIdentities );

        fCallDevice = true;
        return;
    }
    else {
        GetReorderedSubsetIndices(SurfaceIndexSet, fSubsetIdentities, SetSize);
        fCachedSubsetFieldAndPotential = fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, SetSize, aPosition);
        fCallDevice = false;
    }

}

template <class Integrator>
double KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::RetrievePotential() const
{
    if( fCallDevice ) {
        CU_TYPE potential = 0.;

        cudaMemcpy( fHostPotential, fDevicePotential, fCachedNWorkgroups * sizeof(CU_TYPE), cudaMemcpyDeviceToHost );

        for( unsigned int i=0; i<fCachedNWorkgroups; i++ ) {
            potential += fHostPotential[i];
        }

        if( potential != potential ) {
            GetReorderedSubsetIndices(fCachedSurfaceIndexSet, fSubsetIdentities, fCachedSubsetSize);
            return fStandardSolver.Potential(fSubsetIdentities, fCachedSubsetSize, fCachedPosition);
        }

        return potential;
    }
    else {
        return fCachedSubsetPotential;
    }

}

template <class Integrator>
KEMThreeVector KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::RetrieveElectricField() const
{
    if( fCallDevice ) {
        KEMThreeVector eField(0.,0.,0.);

        cudaMemcpy( fHostElectricField, fDeviceElectricField, fCachedNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

        for(unsigned int i=0; i<fCachedNWorkgroups; i++) {
            eField[0] += fHostElectricField[i].x;
            eField[1] += fHostElectricField[i].y;
            eField[2] += fHostElectricField[i].z;
        }

        if( eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2] ) {
            GetReorderedSubsetIndices(fCachedSurfaceIndexSet, fSubsetIdentities, fCachedSubsetSize);
            return fStandardSolver.ElectricField(fSubsetIdentities, fCachedSubsetSize, fCachedPosition);
        }

        return eField;
    }
    else {
        return fCachedSubsetField;
    }

}

template <class Integrator>
std::pair<KEMThreeVector,double> KIntegratingFieldSolver<Integrator,ElectrostaticCUDA>::RetrieveElectricFieldAndPotential() const
{
    if( fCallDevice ) {
        KEMThreeVector eField(0.,0.,0.);
        CU_TYPE potential = 0.;

        cudaMemcpy( fHostElectricField, fDeviceElectricField, fCachedNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyDeviceToHost );

        for(unsigned int i=0; i<fCachedNWorkgroups; i++) {
            eField[0] += fHostElectricFieldAndPotential[i].x;
            eField[1] += fHostElectricFieldAndPotential[i].y;
            eField[2] += fHostElectricFieldAndPotential[i].z;
            potential += fHostElectricFieldAndPotential[i].w;
        }

        if( eField[0] != eField[0] || eField[1] != eField[1] || eField[2] != eField[2] || potential != potential ) {
            GetReorderedSubsetIndices(fCachedSurfaceIndexSet, fSubsetIdentities, fCachedSubsetSize);
            return fStandardSolver.ElectricFieldAndPotential(fSubsetIdentities, fCachedSubsetSize, fCachedPosition);
        }

        return std::make_pair(eField, potential);
    }
    else {
        return fCachedSubsetFieldAndPotential;
    }

}

}

#endif /* KCUDAELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
