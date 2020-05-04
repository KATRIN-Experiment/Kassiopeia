#include "KOpenCLSurfaceContainer.hh"

#include "KOpenCLBufferStreamer.hh"

#include <algorithm>
#include <limits.h>

namespace KEMField
{
KOpenCLSurfaceContainer::KOpenCLSurfaceContainer(const KSurfaceContainer& container) :
    KSortedSurfaceContainer(container),
    KOpenCLData(),
    fNBufferedElements(0),
    fShapeSize(0),
    fBoundarySize(0),
    fBasisSize(0),
    fBufferShapeInfo(NULL),
    fBufferShapeData(NULL),
    fBufferBoundaryInfo(NULL),
    fBufferBoundaryData(NULL),
    fBufferBasisData(NULL)
{
    // Acquire the maximum buffer sizes for the shape, boundary and basis policies
    KSurfaceSize<KShape> shapeSize;
    KSurfaceSize<KBoundary> boundarySize;
    KSurfaceSize<KBasis> basisSize;

    FlagGenerator flagGenerator;

    std::stringstream s;

    std::vector<unsigned short> shapes;
    std::vector<unsigned short> boundaries;

    for (unsigned int i = 0; i < container.NumberOfSurfaceTypes(); i++) {
        KSurfacePrimitive* sP = container.FirstSurfaceType(i);

        shapeSize.Reset();
        shapeSize.SetSurface(sP->GetShape());
        KShapeAction<KSurfaceSize<KShape>>::ActOnShapeType(sP->GetID(), shapeSize);
        if (shapeSize.size() > fShapeSize)
            fShapeSize = shapeSize.size();

        KShapeAction<FlagGenerator>::ActOnShapeType(sP->GetID(), flagGenerator);
        if (std::find(shapes.begin(), shapes.end(), sP->GetID().ShapeID) == shapes.end()) {
            shapes.push_back(sP->GetID().ShapeID);
            s << flagGenerator.GetFlag() << sP->GetID().ShapeID;
        }

        boundarySize.Reset();
        boundarySize.SetSurface(sP->GetBoundary());
        KBoundaryAction<KSurfaceSize<KBoundary>>::ActOnBoundaryType(sP->GetID(), boundarySize);
        if (boundarySize.size() > fBoundarySize)
            fBoundarySize = boundarySize.size();

        KBoundaryAction<FlagGenerator>::ActOnBoundaryType(sP->GetID(), flagGenerator);
        if (std::find(boundaries.begin(), boundaries.end(), sP->GetID().BoundaryID) == boundaries.end()) {
            boundaries.push_back(sP->GetID().BoundaryID);
            s << flagGenerator.GetFlag() << sP->GetID().BoundaryID;
        }

        basisSize.Reset();
        basisSize.SetSurface(sP->GetBasis());
        KBasisAction<KSurfaceSize<KBasis>>::ActOnBasisType(sP->GetID(), basisSize);
        if (basisSize.size() > fBasisSize)
            fBasisSize = basisSize.size();
    }

    s << " -D SHAPESIZE=" << fShapeSize;
    s << " -D BOUNDARYSIZE=" << fBoundarySize;
    s << " -D BASISSIZE=" << fBasisSize;
    fOpenCLFlags = s.str();
}

KOpenCLSurfaceContainer::~KOpenCLSurfaceContainer()
{
    if (fBufferShapeInfo)
        delete fBufferShapeInfo;
    if (fBufferShapeData)
        delete fBufferShapeData;
    if (fBufferBoundaryInfo)
        delete fBufferBoundaryInfo;
    if (fBufferBoundaryData)
        delete fBufferBoundaryData;
    if (fBufferBasisData)
        delete fBufferBasisData;
}

void KOpenCLSurfaceContainer::BuildOpenCLObjects()
{
    // First, we fill a vector with shape data
    unsigned int nDummy = 0;
    unsigned int tmp = fNLocal - (size() % fNLocal);
    if (tmp != fNLocal)
        nDummy += tmp;

    fNBufferedElements = size() + nDummy;

    fShapeInfo.resize(fNBufferedElements, -1);

    fShapeData.resize(fShapeSize * fNBufferedElements, 0.);

    KOpenCLBufferPolicyStreamer<KShape> shapeStreamer;
    shapeStreamer.SetBufferSize(fShapeSize);
    shapeStreamer.SetBuffer(&fShapeData[0]);

    for (unsigned int i = 0; i < size(); i++) {
        fShapeInfo[i] = at(i)->GetID().ShapeID;
        shapeStreamer.SetSurfacePolicy(at(i)->GetShape());
        KShapeAction<KOpenCLBufferPolicyStreamer<KShape>>::ActOnShapeType(at(i)->GetID(), shapeStreamer);
    }

    // Next, we fill a vector with boundary information
    fBoundaryInfo.resize(3 * NUniqueBoundaries() + 2);
    fBoundaryInfo[0] = size();
    fBoundaryInfo[1] = NUniqueBoundaries();


    for (unsigned int i = 0; i < NUniqueBoundaries(); i++) {
        fBoundaryInfo[2 + i * 3] = size(i);
        fBoundaryInfo[2 + i * 3 + 1] = BoundaryType(i);
        fBoundaryInfo[2 + i * 3 + 2] = IndexOfFirstSurface(i);
    }

    // Next, we fill a vector with the actual boundary data
    fBoundaryData.resize(fBoundarySize * NUniqueBoundaries());
    KOpenCLBufferPolicyStreamer<KBoundary> boundaryStreamer;
    boundaryStreamer.SetBufferSize(fBoundarySize);
    boundaryStreamer.SetBuffer(&fBoundaryData[0]);

    for (unsigned int i = 0; i < NUniqueBoundaries(); i++) {
        unsigned int index = IndexOfFirstSurface(i);
        boundaryStreamer.SetSurfacePolicy(at(index)->GetBoundary());
        KBoundaryAction<KOpenCLBufferPolicyStreamer<KBoundary>>::ActOnBoundaryType(at(index)->GetID(),
                                                                                   boundaryStreamer);
    }

    // Finally, we fill a vector with the basis data
    fBasisData.resize(fBasisSize * fNBufferedElements, 0.);

    KOpenCLBufferPolicyStreamer<KBasis> basisStreamer;
    basisStreamer.SetBufferSize(fBasisSize);
    basisStreamer.SetBuffer(&fBasisData[0]);

    for (unsigned int i = 0; i < size(); i++) {
        basisStreamer.SetSurfacePolicy(at(i)->GetBasis());
        KBasisAction<KOpenCLBufferPolicyStreamer<KBasis>>::ActOnBasisType(at(i)->GetID(), basisStreamer);
    }


    // Now that the data is in array form, we can construct the buffers
    fBufferShapeInfo = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                      CL_MEM_READ_ONLY,
                                      fShapeInfo.size() * sizeof(cl_short));

    fBufferShapeData = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                      CL_MEM_READ_ONLY,
                                      fShapeData.size() * sizeof(CL_TYPE));


    fBufferBoundaryInfo = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                         CL_MEM_READ_ONLY,
                                         fBoundaryInfo.size() * sizeof(cl_int));

    fBufferBoundaryData = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                         CL_MEM_READ_ONLY,
                                         fBasisData.size() * sizeof(CL_TYPE));

    fBufferBasisData = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                      CL_MEM_WRITE_ONLY,
                                      fBasisData.size() * sizeof(CL_TYPE));

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferShapeInfo,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fShapeInfo.size() * sizeof(cl_short),
                                                                   &fShapeInfo[0]);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferShapeData,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fShapeData.size() * sizeof(CL_TYPE),
                                                                   &fShapeData[0]);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferBoundaryInfo,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fBoundaryInfo.size() * sizeof(cl_int),
                                                                   &fBoundaryInfo[0]);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferBoundaryData,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fBoundaryData.size() * sizeof(CL_TYPE),
                                                                   &fBoundaryData[0]);


    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBufferBasisData,
                                                                   CL_TRUE,
                                                                   0,
                                                                   fBasisData.size() * sizeof(CL_TYPE),
                                                                   &fBasisData[0]);

    fIsConstructed = true;

    // size_t totalGlbMemInBytes = (fShapeInfo.size() * sizeof(cl_short) +
    // 				 fShapeData.size() * sizeof(CL_TYPE) +
    // 				 fBoundaryInfo.size() * sizeof(cl_int) +
    // 				 fBoundaryData.size() * sizeof(CL_TYPE) +
    // 				 fBasisData.size() * sizeof(CL_TYPE));

    // size_t totalAvailableGlbMemInBytes = KOpenCLInterface::GetInstance()->
    //   GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

    // double bytesInAMegabyte = 1048576.;

    // KEMField::cout<<"Using "<<totalGlbMemInBytes/bytesInAMegabyte<<" of "<<totalAvailableGlbMemInBytes/bytesInAMegabyte<<" MB ("<<((double)totalGlbMemInBytes)/totalAvailableGlbMemInBytes*100.<<" %)"<<KEMField::endl;

    // // let's print the buffers!
    // std::cout<<"surfaces:"<<std::endl;
    // for (unsigned int i=0;i<size();i++)
    //   KEMField::cout<<*(at(i))<<KEMField::endl;
    // std::cout<<""<<std::endl;
    // std::cout<<"shape data:"<<std::endl;
    // for (unsigned int i = 0;i<fShapeData.size();++i)
    // {
    //   if (i%fShapeSize == 0) std::cout<<i/fShapeSize<<":"<<std::endl;
    //   std::cout<<"  "<<i%fShapeSize<<": "<<fShapeData.at(i)<<std::endl;
    // }
    // std::cout<<""<<std::endl;
    // std::cout<<"boundary info:"<<std::endl;
    // for (unsigned int i = 0;i<fBoundaryInfo.size();++i)
    //   std::cout<<i<<" "<<fBoundaryInfo.at(i)<<std::endl;
    // std::cout<<""<<std::endl;
    // std::cout<<"boundary data:"<<std::endl;
    // for (unsigned int i = 0;i<fBoundaryData.size();++i)
    //   std::cout<<i<<" "<<fBoundaryData.at(i)<<std::endl;
    // std::cout<<""<<std::endl;
    // std::cout<<"basis data:"<<std::endl;
    // for (unsigned int i = 0;i<fBasisData.size();++i)
    //   std::cout<<i<<" "<<fBasisData.at(i)<<std::endl;
    // std::cout<<""<<std::endl;
}

void KOpenCLSurfaceContainer::ReadBasisData()
{
    cl::Event event;
    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fBufferBasisData,
                                                                  CL_TRUE,
                                                                  0,
                                                                  fBasisData.size() * sizeof(CL_TYPE),
                                                                  &fBasisData[0],
                                                                  NULL,
                                                                  &event);
    event.wait();

    KOpenCLBufferPolicyStreamer<KBasis> basisStreamer;
    basisStreamer.SetToRead();
    basisStreamer.SetBufferSize(fBasisSize);
    basisStreamer.SetBuffer(&fBasisData[0]);

    for (unsigned int i = 0; i < size(); i++) {
        basisStreamer.SetSurfacePolicy(at(i)->GetBasis());
        KBasisAction<KOpenCLBufferPolicyStreamer<KBasis>>::ActOnBasisType(at(i)->GetID(), basisStreamer);
    }
}

}  // namespace KEMField
