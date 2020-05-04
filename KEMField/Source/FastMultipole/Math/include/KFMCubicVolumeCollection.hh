#ifndef __KFMCubicVolumeCollection_H__
#define __KFMCubicVolumeCollection_H__

#include "KFMCube.hh"

#include <vector>

namespace KEMField
{

/**
*
*@file KFMCubicVolumeCollection.hh
*@class KFMCubicVolumeCollection
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Nov  3 12:11:44 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM> class KFMCubicVolumeCollection
{
  public:
    KFMCubicVolumeCollection()
    {
        fNCubes = 0;
        fCubes.clear();
    };

    virtual ~KFMCubicVolumeCollection(){};

    void Clear()
    {
        fCubes.clear();
        fNCubes = 0;
    }

    void AddCube(KFMCube<NDIM>* cube)
    {
        bool is_present = false;
        for (unsigned int i = 0; i < fNCubes; i++) {
            if (fCubes[i] == cube) {
                is_present = true;
            }
        }

        if (!is_present) {
            fCubes.push_back(cube);
            fNCubes = fCubes.size();
        }
    }


    //navigation
    bool PointIsInside(const double* p) const
    {
        for (unsigned int i = 0; i < fNCubes; i++) {
            if (fCubes[i]->PointIsInside(p)) {
                return true;
            }
        }
        return false;
    }


  protected:
    /* data */
    unsigned int fNCubes;
    std::vector<KFMCube<NDIM>*> fCubes;
};

}  // namespace KEMField

#endif /* __KFMCubicVolumeCollection_H__ */
