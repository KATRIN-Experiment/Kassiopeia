#include "KFMElectrostaticSurfaceConverter.hh"

namespace KEMField
{

void KFMElectrostaticSurfaceConverter::SetSurfaceContainer(const KSurfaceContainer* container)
{
    fSurfaceContainer = container;
    fContainerIsSorted = false;
}

void KFMElectrostaticSurfaceConverter::SetSortedSurfaceContainer(const KSortedSurfaceContainer* container)
{
    fSortedSurfaceContainer = container;
    fContainerIsSorted = true;
}


void KFMElectrostaticSurfaceConverter::SetElectrostaticElementContainer(KFMElectrostaticElementContainerBase<3,1>* container)
{
    fElectrostaticElementContainer = container;
}


void
KFMElectrostaticSurfaceConverter::Extract()
{
    fElectrostaticElementContainer->Clear();

    unsigned int n_elements;

    if(fContainerIsSorted)
    {
        n_elements = fSortedSurfaceContainer->size();
    }
    else
    {
        n_elements = fSurfaceContainer->size();
    }

    int count = 0;
    for(unsigned int i=0; i<n_elements; i++)
    {
        if(fContainerIsSorted)
        {
            fSortedSurfaceContainer->at(i)->Accept(fPointCloudGenerator);
            fSortedSurfaceContainer->at(i)->Accept(fAspectRatioExtractor);
        }
        else
        {
            fSurfaceContainer->at(i)->Accept(fPointCloudGenerator);
            fSurfaceContainer->at(i)->Accept(fAspectRatioExtractor);
        }


        if( fPointCloudGenerator.IsRecognizedType() ) //surface is a triange/rectangle/wire
        {
            fTempPointCloud = fPointCloudGenerator.GetPointCloud();

            if( fTempPointCloud.GetNPoints() != 0 )
            {
                //set the aspect_ratio
                fTempElement.SetAspectRatio(fAspectRatioExtractor.GetAspectRatio());

                //set the point cloud data
                fTempElement.SetPointCloud( fTempPointCloud );

                //compute and set the bounding ball
                KFMBall<3> bball = fBoundingBallGenerator.Convert(&fTempPointCloud);

                //less optimal way of computing bounding sphere (but might be more numerically stable)
                // bball.SetCenter(fTempPointCloud.GetCentroid());
                // bball.SetRadius(fTempPointCloud.GetRadiusAboutCentroid());

                fTempElement.SetBoundingBall( bball );

                //extract the element centroid
                KPosition centroid;
                if(fContainerIsSorted)
                {
                    centroid = fSortedSurfaceContainer->at(i)->GetShape()->Centroid();
                }
                else
                {
                    centroid = fSurfaceContainer->at(i)->GetShape()->Centroid();
                }

                fCentroid[0] = centroid[0];
                fCentroid[1] = centroid[1];
                fCentroid[2] = centroid[2];
                fTempElement.SetCentroid(fCentroid);


                //extract the basis data
                double area = 0.0;
                if(fContainerIsSorted)
                {
                    fSortedSurfaceContainer->at(i)->Accept(fBasisExtractor);
                    area = fSortedSurfaceContainer->at(i)->GetShape()->Area();
                }
                else
                {
                    fSurfaceContainer->at(i)->Accept(fBasisExtractor);
                    area = fSurfaceContainer->at(i)->GetShape()->Area();
                }

                //because the multipole library treats wires as 1-d elements
                //we only store the total charge of an element, and recompute the charge
                //density during the multipole calculation
                KFMBasisData<1> basis = fBasisExtractor.GetBasisData();
                basis[0] = area*basis[0];

                fTempElement.SetBasisData(basis);

                //set the reference ID
                fTempElement.SetIdentityPair( KFMIdentityPair(count, i) );

                fElectrostaticElementContainer->AddElectrostaticElement(fTempElement);
            }

            count++;
        }
    }
}

void
KFMElectrostaticSurfaceConverter::UpdateBasisData()
{
    unsigned int n_elements = fElectrostaticElementContainer->GetNElements();

    for(unsigned int i=0; i<n_elements; i++)
    {
        //retrieve the elements index in the surface container
        //unsigned int id = i;fElectrostaticElementContainer->GetIdentityPair(i)->GetMappedID();

        //extract the basis data
        double area = 0.0;
        if(fContainerIsSorted)
        {
            fSortedSurfaceContainer->at(i)->Accept(fBasisExtractor);
            area = fSortedSurfaceContainer->at(i)->GetShape()->Area();
        }
        else
        {
            fSurfaceContainer->at(i)->Accept(fBasisExtractor);
            area = fSurfaceContainer->at(i)->GetShape()->Area();
        }
        KFMBasisData<1> basis = fBasisExtractor.GetBasisData();
        KFMBasisData<1>* basis_ptr = fElectrostaticElementContainer->GetBasisData(i);
        (*basis_ptr)[0] = area*basis[0];
    }
}


void
KFMElectrostaticSurfaceConverter::UpdateBasisData(const KVector<double>& x)
{
    //we expect the update vector to be the charge densities
    //we then convert this to total charge

    unsigned int n_elements = fElectrostaticElementContainer->GetNElements();

    for(unsigned int i=0; i<n_elements; i++)
    {
        //retrieve the elements index in the surface container
        //unsigned int id = i;fElectrostaticElementContainer->GetIdentityPair(i)->GetMappedID();

        //extract the basis data

        //extract the basis data
        double area = 0.0;
        if(fContainerIsSorted)
        {
            area = fSortedSurfaceContainer->at(i)->GetShape()->Area();
        }
        else
        {
            area = fSurfaceContainer->at(i)->GetShape()->Area();
        }

        double cd = x(i);
        KFMBasisData<1>* basis_ptr = fElectrostaticElementContainer->GetBasisData(i);
        (*basis_ptr)[0] = area*cd;
    }

}



}//end of KEMField
