#include "KCUDAElectrostaticIntegratingFieldSolver.hh"

#include <limits.h>
#include <fstream>
#include <sstream>

#define MAX_SUBSET_SIZE 10000
#define MIN_SUBSET_SIZE 16

namespace KEMField
{
  KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::
  KIntegratingFieldSolver(KCUDASurfaceContainer& container,
			  KCUDAElectrostaticBoundaryIntegrator& integrator) :
			  KCUDAAction(container),
			  fContainer(container),
			  fStandardSolver(container.GetSurfaceContainer(),fStandardIntegrator),
			  fHostPotential(NULL),
			  fHostElectricField(NULL),
			  fMaxSubsetSize(MAX_SUBSET_SIZE),
			  fMinSubsetSize(MIN_SUBSET_SIZE),
			  fDeviceElementIdentities(NULL)
  {
    fSubsetIdentities = new unsigned int[MAX_SUBSET_SIZE];
	fReorderedSubsetIdentities = new unsigned int[MAX_SUBSET_SIZE];
  }


  KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::
  KIntegratingFieldSolver(KCUDASurfaceContainer& container,
			  KCUDAElectrostaticBoundaryIntegrator& integrator,
              unsigned int max_subset_size,
			  unsigned int min_subset_size) :
              KCUDAAction(container),
              fContainer(container),
              fStandardSolver(container.GetSurfaceContainer(),fStandardIntegrator),
              fHostPotential(NULL),
              fHostElectricField(NULL),
              fMaxSubsetSize(max_subset_size),
              fMinSubsetSize(min_subset_size),
              fDeviceElementIdentities(NULL)
  {
    if( fMaxSubsetSize == 0 ) {fMaxSubsetSize = MAX_SUBSET_SIZE;};
    fSubsetIdentities = new unsigned int[fMaxSubsetSize];
	fReorderedSubsetIdentities = new unsigned int[fMaxSubsetSize];
  }



  KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::
  ~KIntegratingFieldSolver()
  {
    if( fHostPotential ) delete fHostPotential;
    if( fHostElectricField ) delete fHostElectricField;
    if( fDeviceElementIdentities ) cudaFree(fDeviceElementIdentities);
    delete[] fSubsetIdentities;
    delete[] fReorderedSubsetIdentities;
  }

  void KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::ConstructCUDAKernels() const
  {
    fNLocal = 1024;

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

  void KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::AssignDeviceMemory() const
  {
    fNGlobal = fContainer.GetNBufferedElements();
    fNLocal = fContainer.GetMinimumWorkgroupSizeForKernels();
    fNWorkgroups = fContainer.GetNBufferedElements()/fNLocal;

    cudaMalloc( (void**) &fDeviceP, 3*sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDevicePotential, fNWorkgroups*sizeof(CU_TYPE) );
    cudaMalloc( (void**) &fDeviceElectricField, fNWorkgroups*sizeof(CU_TYPE4) );

    fHostPotential = new CU_TYPE[fNWorkgroups];
    fHostElectricField = new CU_TYPE4[fNWorkgroups];

    cudaMemcpy( fDevicePotential, fHostPotential, fNWorkgroups*sizeof(CU_TYPE), cudaMemcpyHostToDevice );
    cudaMemcpy( fDeviceElectricField, fHostElectricField, fNWorkgroups*sizeof(CU_TYPE4), cudaMemcpyHostToDevice );

    //create the element id buffer
    cudaMalloc( (void**) &fDeviceElementIdentities, fMaxSubsetSize*sizeof(unsigned int) );
  }

  double KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::Potential(const KPosition& aPosition) const
  {
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

    if( KEMFIELD_OCCUPANCYAPI ) {
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, PotentialKernel, 0, 0);
        std::cout << "[PotentialKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[PotentialKernel] Set block size: " << fNLocal << std::endl;
    }

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

  KEMThreeVector KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::ElectricField(const KPosition& aPosition) const
  {
    CU_TYPE P[3] = {aPosition[0],aPosition[1],aPosition[2]};
    cudaMemcpy( fDeviceP, P, 3*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

    if( KEMFIELD_OCCUPANCYAPI ) {
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ElectricFieldKernel, 0, 0);
        std::cout << "[ElectricFieldKernel] Suggested block size: " << blockSize << std::endl;
        std::cout << "[ElectricFieldKernel] Set block size: " << fNLocal << std::endl;
    }

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

    double KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
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

            if( KEMFIELD_OCCUPANCYAPI ) {
                int blockSize;   // The launch configurator returned block size
                int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
                cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetPotentialKernel, 0, 0);
                std::cout << "[SubsetPotentialKernel] Suggested block size: " << blockSize << std::endl;
                std::cout << "[SubsetPotentialKernel] Set block size: " << local << std::endl;
            }

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

    KEMThreeVector KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
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

            if( KEMFIELD_OCCUPANCYAPI ) {
                int blockSize;   // The launch configurator returned block size
                int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
                cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetElectricFieldKernel, 0, 0);
                std::cout << "[SubsetElectricFieldKernel] Suggested block size: " << blockSize << std::endl;
                std::cout << "[SubsetElectricFieldKernel] Set block size: " << local << std::endl;
            }

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


    void KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::GetReorderedSubsetIndices(const unsigned int* sorted_container_ids, unsigned int* normal_container_ids, unsigned int size) const
    {
        for( unsigned int i=0; i<size; i++ ) {
            normal_container_ids[i] = fContainer.GetNormalIndexFromSortedIndex(sorted_container_ids[i]);
        }
    }

    void KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::DispatchPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
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

            if( KEMFIELD_OCCUPANCYAPI ) {
                int blockSize;   // The launch configurator returned block size
                int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
                cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetPotentialKernel, 0, 0);
                std::cout << "[SubsetPotentialKernel] Suggested block size: " << blockSize << std::endl;
                std::cout << "[SubsetPotentialKernel] Set block size: " << local << std::endl;
            }

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

    void KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::DispatchElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& aPosition) const
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

            if( KEMFIELD_OCCUPANCYAPI ) {
                int blockSize;   // The launch configurator returned block size
                int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
                cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, SubsetElectricFieldKernel, 0, 0);
                std::cout << "[SubsetElectricFieldKernel] Suggested block size: " << blockSize << std::endl;
                std::cout << "[SubsetElectricFieldKernel] Set block size: " << local << std::endl;
            }

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

    double KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::RetrievePotential() const
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

    KEMThreeVector
    KIntegratingFieldSolver<KCUDAElectrostaticBoundaryIntegrator>::RetrieveElectricField() const
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



}//end KEMField namespace
