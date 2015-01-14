#ifndef KFMElectrostaticBasisDataContainer_HH__
#define KFMElectrostaticBasisDataContainer_HH__

#include "KFMObjectContainer.hh"

#include "KSurfaceTypes.hh"
#include "KSurfaceContainer.hh"
#include "KSortedSurfaceContainer.hh"

#include "KBasis.hh"
#include "KElectrostaticBasis.hh"

#include "KFMBasisData.hh"
#include "KFMElectrostaticBasisDataExtractor.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticBasisDataContainer.hh
*@class KFMElectrostaticBasisDataContainer
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr 10 12:26:20 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

typedef KFMObjectContainer< KFMBasisData<1> > KFMElectrostaticBasisDataContainerPiecewiseConstant;

class KFMElectrostaticBasisDataContainer: public KFMElectrostaticBasisDataContainerPiecewiseConstant
{
    public:

        KFMElectrostaticBasisDataContainer(const KSurfaceContainer& container):fSurfaceContainer(&container),fSortedSurfaceContainer(NULL)
        {
            fContainerIsSorted = false;
        };

        KFMElectrostaticBasisDataContainer(const KSortedSurfaceContainer& container):fSurfaceContainer(NULL), fSortedSurfaceContainer(&container)
        {
            fContainerIsSorted = true;
        };

        virtual ~KFMElectrostaticBasisDataContainer(){};

        virtual unsigned int GetNObjects() const
        {
            if(fContainerIsSorted)
            {
                return fSortedSurfaceContainer->size();
            }
            else
            {
                return fSurfaceContainer->size();
            }
        };

        virtual void AddObject(const KFMBasisData<1>& /*obj*/)
        {
            //warning...cannot add object to a virtual container
        }

        virtual KFMBasisData<1>* GetObjectWithID(const unsigned int& id)
        {
            if(fContainerIsSorted)
            {
                fSortedSurfaceContainer->at(id)->Accept(fBasisExtractor);
                double area = fSortedSurfaceContainer->at(id)->GetShape()->Area();

                //because the multipole library treats wires as 1-d elements
                //we only store the total charge of an element, and recompute the charge
                //density during the multipole calculation, as a linear or areal charge density
                fCurrentBasisData = fBasisExtractor.GetBasisData();
                fCurrentBasisData[0] = area*fCurrentBasisData[0];
                return &fCurrentBasisData;
            }
            else
            {
                fSurfaceContainer->at(id)->Accept(fBasisExtractor);
                double area = fSurfaceContainer->at(id)->GetShape()->Area();

                //because the multipole library treats wires as 1-d elements
                //we only store the total charge of an element, and recompute the charge
                //density during the multipole calculation, as a linear or areal charge density
                fCurrentBasisData = fBasisExtractor.GetBasisData();
                fCurrentBasisData[0] = area*fCurrentBasisData[0];
                return &fCurrentBasisData;
            }
        }

        virtual const KFMBasisData<1>* GetObjectWithID(const unsigned int& id) const
        {
            if(fContainerIsSorted)
            {
                fSortedSurfaceContainer->at(id)->Accept(fBasisExtractor);
                double area = fSortedSurfaceContainer->at(id)->GetShape()->Area();

                //because the multipole library treats wires as 1-d elements
                //we only store the total charge of an element, and recompute the charge
                //density during the multipole calculation, as a linear or areal charge density
                fCurrentBasisData = fBasisExtractor.GetBasisData();
                fCurrentBasisData[0] = area*fCurrentBasisData[0];
                return &fCurrentBasisData;
            }
            else
            {
                fSurfaceContainer->at(id)->Accept(fBasisExtractor);
                double area = fSurfaceContainer->at(id)->GetShape()->Area();

                //because the multipole library treats wires as 1-d elements
                //we only store the total charge of an element, and recompute the charge
                //density during the multipole calculation, as a linear or areal charge density
                fCurrentBasisData = fBasisExtractor.GetBasisData();
                fCurrentBasisData[0] = area*fCurrentBasisData[0];
                return &fCurrentBasisData;
            }
        }

        virtual void DeleteAllObjects(){;}; //does nothing, no objects to delete


    private:

        const KSurfaceContainer* fSurfaceContainer;
        const KSortedSurfaceContainer* fSortedSurfaceContainer;
        bool fContainerIsSorted;

        mutable KFMElectrostaticBasisDataExtractor fBasisExtractor;
        mutable KFMBasisData<1> fCurrentBasisData;


};

}



#endif /* KFMElectrostaticBasisDataContainer_H__ */
