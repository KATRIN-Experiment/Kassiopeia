#include "KFMElectrostaticElementContainerFlyweight.hh"
#include "KFMMessaging.hh"

namespace KEMField
{

KFMElectrostaticElementContainerFlyweight::KFMElectrostaticElementContainerFlyweight(const KSurfaceContainer& container):
fSurfaceContainer(&container),
fSortedSurfaceContainer(NULL),
fPointCloudContainer(container),
fBoundingBallContainer(container)
{
    fContainerIsSorted = false;

    unsigned int n_elements = fSurfaceContainer->size();

    fAspectRatio.resize(n_elements);

    //extract the basis data
    KFMElectrostaticBasisDataExtractor bd_extractor;

    //extract the aspect ratio
    KFMElementAspectRatioExtractor ar_extractor;


    for(unsigned int i=0; i<n_elements; i++)
    {
        fSurfaceContainer->at(i)->Accept(bd_extractor);
        double area = fSurfaceContainer->at(i)->GetShape()->Area();

        fSurfaceContainer->at(i)->Accept(ar_extractor);
        fAspectRatio[i] = ar_extractor.GetAspectRatio();

        //because the multipole library treats wires as 1-d elements
        //we only store the total charge of an element, and recompute the charge
        //density during the multipole calculation
        KFMBasisData<1> basis = bd_extractor.GetBasisData();
        basis[0] = area*basis[0];

        fBasisContainer.AddObject(basis);
    }
};



KFMElectrostaticElementContainerFlyweight::KFMElectrostaticElementContainerFlyweight(const KSortedSurfaceContainer& container):
fSurfaceContainer(NULL),
fSortedSurfaceContainer(&container),
fPointCloudContainer(container),
fBoundingBallContainer(container)
{
    fContainerIsSorted = true;

    //extract the basis data
    KFMElectrostaticBasisDataExtractor bd_extractor;

    //extract the aspect ratio
    KFMElementAspectRatioExtractor ar_extractor;

    unsigned int n_elements = fSortedSurfaceContainer->size();
    for(unsigned int i=0; i<n_elements; i++)
    {
        fSurfaceContainer->at(i)->Accept(ar_extractor);
        fAspectRatio[i] = ar_extractor.GetAspectRatio();

        fSortedSurfaceContainer->at(i)->Accept(bd_extractor);
        double area = fSortedSurfaceContainer->at(i)->GetShape()->Area();

        //because the multipole library treats wires as 1-d elements
        //we only store the total charge of an element, and recompute the charge
        //density during the multipole calculation
        KFMBasisData<1> basis = bd_extractor.GetBasisData();
        basis[0] = area*basis[0];

        fBasisContainer.AddObject(basis);
    }
};


unsigned int
KFMElectrostaticElementContainerFlyweight::GetNElements() const
{
    if(fContainerIsSorted)
    {
        return fSortedSurfaceContainer->size();
    }
    else
    {
        return fSurfaceContainer->size();
    }
}

void
KFMElectrostaticElementContainerFlyweight::AddElectrostaticElement(const KFMElectrostaticElement< 3, 1 >& /*elem*/)
{
    //warning, cannot add objects to  container
}

KFMElectrostaticElement<3, 1>
KFMElectrostaticElementContainerFlyweight::GetElectrostaticElement(unsigned int id)
{
    KFMElectrostaticElement<3, 1> elem;
    elem.SetPointCloud(*(GetPointCloud(id)));
    elem.SetBasisData(*(GetBasisData(id)));
    elem.SetBoundingBall(*(GetBoundingBall(id)));
    return elem;
}

KFMPointCloud<3>*
KFMElectrostaticElementContainerFlyweight::GetPointCloud(unsigned int id){return fPointCloudContainer.GetObjectWithID(id);};

KFMBall<3>*
KFMElectrostaticElementContainerFlyweight::GetBoundingBall(unsigned int id){return fBoundingBallContainer.GetObjectWithID(id);};

KFMBasisData<1>*
KFMElectrostaticElementContainerFlyweight::GetBasisData(unsigned int id){return fBasisContainer.GetObjectWithID(id);};

const KFMPointCloud<3>*
KFMElectrostaticElementContainerFlyweight::GetPointCloud(unsigned int id) const {return fPointCloudContainer.GetObjectWithID(id);};

const KFMBall<3>*
KFMElectrostaticElementContainerFlyweight::GetBoundingBall(unsigned int id) const {return fBoundingBallContainer.GetObjectWithID(id);};

const KFMBasisData<1>*
KFMElectrostaticElementContainerFlyweight::GetBasisData(unsigned int id) const {return fBasisContainer.GetObjectWithID(id);};

double
KFMElectrostaticElementContainerFlyweight::GetAspectRatio(unsigned int id) const
{
    return fAspectRatio[id];
}


KFMObjectContainer< KFMPointCloud<3> >*
KFMElectrostaticElementContainerFlyweight::GetPointCloudContainer(){return &fPointCloudContainer;};

KFMObjectContainer< KFMBall<3> >*
KFMElectrostaticElementContainerFlyweight::GetBoundingBallContainer(){return &fBoundingBallContainer;};

KFMObjectContainer< KFMBasisData<1> >*
KFMElectrostaticElementContainerFlyweight::GetChargeDensityContainer(){return &fBasisContainer;};

const KFMObjectContainer< KFMPointCloud<3> >*
KFMElectrostaticElementContainerFlyweight::GetPointCloudContainer() const {return &fPointCloudContainer;};

const KFMObjectContainer< KFMBall<3> >*
KFMElectrostaticElementContainerFlyweight::GetBoundingBallContainer() const {return &fBoundingBallContainer;};

const KFMObjectContainer< KFMBasisData<1> >*
KFMElectrostaticElementContainerFlyweight::GetChargeDensityContainer() const {return &fBasisContainer;};


};
