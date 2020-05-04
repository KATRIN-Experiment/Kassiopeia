#ifndef KFMBoundingBallContainer_HH__
#define KFMBoundingBallContainer_HH__

#include "KFMBall.hh"
#include "KFMObjectContainer.hh"
#include "KFMPointCloud.hh"
#include "KFMPointCloudToBoundingBallConverter.hh"
#include "KFMSurfaceToPointCloudConverter.hh"
#include "KSortedSurfaceContainer.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"

namespace KEMField
{

/*
*
*@file KFMBoundingBallContainer.hh
*@class KFMBoundingBallContainer
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr 10 12:26:20 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

typedef KFMObjectContainer<KFMBall<3>> KFMBoundingBallContainer3D;

class KFMBoundingBallContainer : public KFMBoundingBallContainer3D
{
  public:
    KFMBoundingBallContainer(const KSurfaceContainer& container) :
        fSurfaceContainer(&container),
        fSortedSurfaceContainer(NULL)
    {
        fContainerIsSorted = false;
    };

    KFMBoundingBallContainer(const KSortedSurfaceContainer& container) :
        fSurfaceContainer(NULL),
        fSortedSurfaceContainer(&container)
    {
        fContainerIsSorted = true;
    };

    virtual ~KFMBoundingBallContainer(){};

    virtual unsigned int GetNObjects() const
    {
        if (fContainerIsSorted) {
            return fSortedSurfaceContainer->size();
        }
        else {
            return fSurfaceContainer->size();
        }
    };

    virtual void AddObject(const KFMBall<3>& /*obj*/)
    {
        //warning...cannot add object to a virtual container
    }

    virtual KFMBall<3>* GetObjectWithID(const unsigned int& id)
    {
        if (fContainerIsSorted) {
            fSortedSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                fCurrentBoundingBall = fBoundingBallGenerator.Convert(&fCurrentPointCloud);
                return &fCurrentBoundingBall;
            }
            else {
                return NULL;
            }
        }
        else {
            fSurfaceContainer->at(id)->Accept(fPointCloudGenerator);

            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                fCurrentBoundingBall = fBoundingBallGenerator.Convert(&fCurrentPointCloud);
                return &fCurrentBoundingBall;
            }
            else {
                return NULL;
            }
        }
    }

    virtual const KFMBall<3>* GetObjectWithID(const unsigned int& id) const
    {
        if (fContainerIsSorted) {
            fSortedSurfaceContainer->at(id)->Accept(fPointCloudGenerator);
            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                fCurrentBoundingBall = fBoundingBallGenerator.Convert(&fCurrentPointCloud);
                return &fCurrentBoundingBall;
            }
            else {
                return NULL;
            }
        }
        else {
            fSurfaceContainer->at(id)->Accept(fPointCloudGenerator);
            if (fPointCloudGenerator.IsRecognizedType())  //surface is a triange/rectangle/wire
            {
                fCurrentPointCloud = fPointCloudGenerator.GetPointCloud();
                fCurrentBoundingBall = fBoundingBallGenerator.Convert(&fCurrentPointCloud);
                return &fCurrentBoundingBall;
            }
            else {
                return NULL;
            }
        }
    }

    virtual void DeleteAllObjects()
    {
        ;
    };  //does nothing, no objects to delete


  private:
    const KSurfaceContainer* fSurfaceContainer;
    const KSortedSurfaceContainer* fSortedSurfaceContainer;
    bool fContainerIsSorted;

    mutable KFMSurfaceToPointCloudConverter fPointCloudGenerator;
    mutable KFMPointCloudToBoundingBallConverter<3> fBoundingBallGenerator;
    mutable KFMPointCloud<3> fCurrentPointCloud;
    mutable KFMBall<3> fCurrentBoundingBall;
};

}  // namespace KEMField


#endif /* KFMBoundingBallContainer_H__ */
