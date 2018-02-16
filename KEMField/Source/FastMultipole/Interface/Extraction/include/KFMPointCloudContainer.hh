#ifndef KFMPointCloudContainer_HH__
#define KFMPointCloudContainer_HH__

#include "KFMObjectContainer.hh"

#include "KFMPointCloud.hh"
#include "KSurfaceTypes.hh"
#include "KSurfaceContainer.hh"
#include "KSortedSurfaceContainer.hh"

#include "KFMSurfaceToPointCloudConverter.hh"

namespace KEMField
{

/*
*
*@file KFMPointCloudContainer.hh
*@class KFMPointCloudContainer
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr 10 12:26:20 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

typedef KFMObjectContainer< KFMPointCloud<3> > KFMPointCloudContainer3D;

class KFMPointCloudContainer: public KFMPointCloudContainer3D
{
    public:

        KFMPointCloudContainer(const KSurfaceContainer& container):fSurfaceContainer(&container),fSortedSurfaceContainer(NULL)
        {
            fContainerIsSorted = false;
        };

        KFMPointCloudContainer(const KSortedSurfaceContainer& container):fSurfaceContainer(NULL), fSortedSurfaceContainer(&container)
        {
            fContainerIsSorted = true;
        };

        virtual ~KFMPointCloudContainer(){};

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

        virtual void AddObject(const KFMPointCloud<3>& /*obj*/)
        {
            //warning...cannot add object to a virtual container
        }

        virtual KFMPointCloud<3>* GetObjectWithID(const unsigned int& id)
        {
            if(fContainerIsSorted)
            {
                fSortedSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

                if( fPointCloudGenerator.IsRecognizedType() ) //surface is a triange/rectangle/wire
                {
                    fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                    return &fCurrentPointCloud;
                }
                else
                {
                    return NULL;
                }
            }
            else
            {
                fSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

                if( fPointCloudGenerator.IsRecognizedType() ) //surface is a triange/rectangle/wire
                {
                    fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                    return &fCurrentPointCloud;
                }
                else
                {
                    return NULL;
                }
            }
        }

        virtual const KFMPointCloud<3>* GetObjectWithID(const unsigned int& id) const
        {
            if(fContainerIsSorted)
            {
                fSortedSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

                if( fPointCloudGenerator.IsRecognizedType() ) //surface is a triange/rectangle/wire
                {
                    fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                    return &fCurrentPointCloud;
                }
                else
                {
                    return NULL;
                }
            }
            else
            {
                fSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

                if( fPointCloudGenerator.IsRecognizedType() ) //surface is a triange/rectangle/wire
                {
                    fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                    return &fCurrentPointCloud;
                }
                else
                {
                    return NULL;
                }
            }
        }

        virtual void DeleteAllObjects(){;}; //does nothing, no objects to delete


    private:

        const KSurfaceContainer* fSurfaceContainer;
        const KSortedSurfaceContainer* fSortedSurfaceContainer;
        bool fContainerIsSorted;
        mutable KFMSurfaceToPointCloudConverter fPointCloudGenerator;
        mutable KFMPointCloud<3> fCurrentPointCloud;

};

}


#endif /* KFMPointCloudContainer_H__ */
