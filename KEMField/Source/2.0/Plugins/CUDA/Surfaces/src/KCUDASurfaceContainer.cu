#include "KCUDASurfaceContainer.hh"

#include <limits.h>
#include <algorithm>

#include <iostream>

#include "KCUDABufferStreamer.hh"

namespace KEMField
{
  KCUDASurfaceContainer::KCUDASurfaceContainer(const KSurfaceContainer& container) :
    KSortedSurfaceContainer(container),
    KCUDAData(),
    fNBufferedElements(0),
    fShapeSize(0),
    fBoundarySize(0),
    fBasisSize(0),
    fDeviceShapeInfo(NULL),
    fDeviceShapeData(NULL),
    fDeviceBoundaryInfo(NULL),
    fDeviceBoundaryData(NULL),
    fDeviceBasisData(NULL)
  {
    // Acquire the maximum buffer sizes for the shape, boundary and basis policies
    KSurfaceSize<KShape> shapeSize;
    KSurfaceSize<KBoundary> boundarySize;
    KSurfaceSize<KBasis> basisSize;

    FlagGenerator flagGenerator;

    std::vector<unsigned short> shapes;
    std::vector<unsigned short> boundaries;

    for (unsigned int i=0;i<container.NumberOfSurfaceTypes();i++)
    {
      KSurfacePrimitive* sP = container.FirstSurfaceType(i);

      // Shape size (maximal)

      shapeSize.Reset();
      shapeSize.SetSurface( sP->GetShape() );
      KShapeAction<KSurfaceSize<KShape> >::ActOnShapeType( sP->GetID(), shapeSize );
      if( shapeSize.size()>fShapeSize ) fShapeSize = shapeSize.size();

	  // Shape type (triangle, rectangle, line segment, conic section)

      KShapeAction<FlagGenerator>::ActOnShapeType( sP->GetID(), flagGenerator );
      if( std::find(shapes.begin(),shapes.end(),sP->GetID().ShapeID) == shapes.end() ) {
        shapes.push_back(sP->GetID().ShapeID);
      }

      // Boundary size (maximal)

      boundarySize.Reset();
      boundarySize.SetSurface( sP->GetBoundary() );
      KBoundaryAction<KSurfaceSize<KBoundary> >::ActOnBoundaryType( sP->GetID(),boundarySize );
      if( boundarySize.size()>fBoundarySize ) fBoundarySize = boundarySize.size();

	  // Boundary type (Dirichlet, Neumann)

      KBoundaryAction<FlagGenerator>::ActOnBoundaryType( sP->GetID(),flagGenerator );
      if( std::find(boundaries.begin(),boundaries.end(),sP->GetID().BoundaryID) == boundaries.end() ) {
        boundaries.push_back(sP->GetID().BoundaryID);
      }

      // Basis size

      basisSize.Reset();
      basisSize.SetSurface(sP->GetBasis());
      KBasisAction<KSurfaceSize<KBasis> >::ActOnBasisType(sP->GetID(),basisSize);
      if( basisSize.size()>fBasisSize ) fBasisSize = basisSize.size();
      
    }
  }

  KCUDASurfaceContainer::~KCUDASurfaceContainer()
  {
    if (fDeviceShapeInfo)    cudaFree(fDeviceShapeInfo);
    if (fDeviceShapeData)    cudaFree(fDeviceShapeData);
    if (fDeviceBoundaryInfo) cudaFree(fDeviceBoundaryInfo);
    if (fDeviceBoundaryData) cudaFree(fDeviceBoundaryData);
    if (fDeviceBasisData)    cudaFree(fDeviceBasisData);
  }

  void KCUDASurfaceContainer::BuildCUDAObjects()
  {
    // First, we fill a vector with shape data
    unsigned int nDummy = 0;
    unsigned int tmp = fNLocal - (size()%fNLocal);
    if (tmp != fNLocal)
      nDummy += tmp;

    fNBufferedElements = size() + nDummy;

    fShapeInfo.resize(fNBufferedElements,-1);

    fShapeData.resize(fShapeSize*fNBufferedElements,0.);

    KCUDABufferPolicyStreamer<KShape> shapeStreamer;
    shapeStreamer.SetBufferSize(fShapeSize);
    shapeStreamer.SetBuffer(&fShapeData[0]);

    for( unsigned int i=0; i<size(); i++ ) {
      fShapeInfo[i] = at(i)->GetID().ShapeID;
      shapeStreamer.SetSurfacePolicy(at(i)->GetShape());
      KShapeAction<KCUDABufferPolicyStreamer<KShape> >::ActOnShapeType(at(i)->GetID(),shapeStreamer);
    }

    // Next, we fill a vector with boundary information
    fBoundaryInfo.resize(3*NUniqueBoundaries()+2);
    fBoundaryInfo[0] = size();
    fBoundaryInfo[1] = NUniqueBoundaries();


    for( unsigned int i=0; i<NUniqueBoundaries(); i++ ) {
      fBoundaryInfo[2 + i*3] = size(i);
      fBoundaryInfo[2 + i*3 + 1] = BoundaryType(i);
      fBoundaryInfo[2 + i*3 + 2] = IndexOfFirstSurface(i);
    }

    // Next, we fill a vector with the actual boundary data
    fBoundaryData.resize(fBoundarySize*NUniqueBoundaries());
    KCUDABufferPolicyStreamer<KBoundary> boundaryStreamer;
    boundaryStreamer.SetBufferSize(fBoundarySize);
    boundaryStreamer.SetBuffer(&fBoundaryData[0]);

    for( unsigned int i=0; i<NUniqueBoundaries(); i++ ) {
      unsigned int index = IndexOfFirstSurface(i);
      boundaryStreamer.SetSurfacePolicy(at(index)->GetBoundary());
      KBoundaryAction<KCUDABufferPolicyStreamer<KBoundary> >::ActOnBoundaryType(at(index)->GetID(),boundaryStreamer);
    }

    // Finally, we fill a vector with the basis data
    fBasisData.resize(fBasisSize*fNBufferedElements,0.);

    KCUDABufferPolicyStreamer<KBasis> basisStreamer;
    basisStreamer.SetBufferSize(fBasisSize);
    basisStreamer.SetBuffer(&fBasisData[0]);

    for( unsigned int i=0;i<size();i++ ) {
      basisStreamer.SetSurfacePolicy(at(i)->GetBasis());
      KBasisAction<KCUDABufferPolicyStreamer<KBasis> >::ActOnBasisType(at(i)->GetID(),basisStreamer);
    }

    // Now that the data is in array form, we can allocate the device memory
    cudaMalloc((void**) &fDeviceShapeInfo, fShapeInfo.size()*sizeof(short));
    cudaMalloc((void**) &fDeviceShapeData, fShapeData.size()*sizeof(CU_TYPE));
    cudaMalloc((void**) &fDeviceBoundaryInfo, fBoundaryInfo.size()*sizeof(int));
    cudaMalloc((void**) &fDeviceBoundaryData, fBasisData.size()*sizeof(CU_TYPE));
    cudaMalloc((void**) &fDeviceBasisData, fBasisData.size()*sizeof(CU_TYPE));

    // Write to device memory
    cudaMemcpy(fDeviceShapeInfo, &fShapeInfo[0], fShapeInfo.size()*sizeof(short), cudaMemcpyHostToDevice );
    cudaMemcpy(fDeviceShapeData, &fShapeData[0], fShapeData.size()*sizeof(CU_TYPE), cudaMemcpyHostToDevice );
    cudaMemcpy(fDeviceBoundaryInfo, &fBoundaryInfo[0], fBoundaryInfo.size()*sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy(fDeviceBoundaryData, &fBoundaryData[0], fBoundaryData.size()*sizeof(CU_TYPE), cudaMemcpyHostToDevice );
    cudaMemcpy(fDeviceBasisData, &fBasisData[0], fBasisData.size()*sizeof(CU_TYPE), cudaMemcpyHostToDevice );

    fIsConstructed = true;
  }

  void KCUDASurfaceContainer::ReadBasisData()
  {
    cudaMemcpy( &fBasisData[0], fDeviceBasisData, fBasisData.size()*sizeof(CU_TYPE), cudaMemcpyDeviceToHost );

    KCUDABufferPolicyStreamer<KBasis> basisStreamer;
    basisStreamer.SetToRead();
    basisStreamer.SetBufferSize(fBasisSize);
    basisStreamer.SetBuffer(&fBasisData[0]);

    for( unsigned int i=0;i<size();i++ ) {
      basisStreamer.SetSurfacePolicy(at(i)->GetBasis());
      KBasisAction<KCUDABufferPolicyStreamer<KBasis> >::ActOnBasisType(at(i)->GetID(),basisStreamer);
    }

  }

}
