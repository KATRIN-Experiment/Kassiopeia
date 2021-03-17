#ifndef KFMPointCloudContainer_HH__
#define KFMPointCloudContainer_HH__

#include "KFMObjectContainer.hh"
#include "KFMPointCloud.hh"
#include "KFMSurfaceToPointCloudConverter.hh"
#include "KSortedSurfaceContainer.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"

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

typedef KFMObjectContainer<KFMPointCloud<3>> KFMPointCloudContainer3D;

class KFMPointCloudContainer : public KFMPointCloudContainer3D
{
  public:
    KFMPointCloudContainer(const KSurfaceContainer& container) :
        fSurfaceContainer(&container),
        fSortedSurfaceContainer(nullptr)
    {
        fContainerIsSorted = false;
    };

    KFMPointCloudContainer(const KSortedSurfaceContainer& container) :
        fSurfaceContainer(nullptr),
        fSortedSurfaceContainer(&container)
    {
        fContainerIsSorted = true;
    };

    ~KFMPointCloudContainer() override = default;
    ;

    unsigned int GetNObjects() const override
    {
        if (fContainerIsSorted) {
            return fSortedSurfaceContainer->size();
        }
        else {
            return fSurfaceContainer->size();
        }
    };

    void AddObject(const KFMPointCloud<3>& /*obj*/) override
    {
        //warning...cannot add object to a virtual container
    }

    KFMPointCloud<3>* GetObjectWithID(const unsigned int& id) override
    {
        if (fContainerIsSorted) {
            fSortedSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                return &fCurrentPointCloud;
            }
            else {
                return nullptr;
            }
        }
        else {
            fSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                return &fCurrentPointCloud;
            }
            else {
                return nullptr;
            }
        }
    }

    const KFMPointCloud<3>* GetObjectWithID(const unsigned int& id) const override
    {
        if (fContainerIsSorted) {
            fSortedSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                return &fCurrentPointCloud;
            }
            else {
                return nullptr;
            }
        }
        else {
            fSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                return &fCurrentPointCloud;
            }
            else {
                return nullptr;
            }
        }
    }

    void DeleteAllObjects() override
    {
        ;
    };  //does nothing, no objects to delete


  private:
    const KSurfaceContainer* fSurfaceContainer;
    const KSortedSurfaceContainer* fSortedSurfaceContainer;
    bool fContainerIsSorted;
    mutable KFMSurfaceToPointCloudConverter fPointCloudGenerator;
    mutable KFMPointCloud<3> fCurrentPointCloud;
};

}  // namespace KEMField


#endif /* KFMPointCloudContainer_H__ */
