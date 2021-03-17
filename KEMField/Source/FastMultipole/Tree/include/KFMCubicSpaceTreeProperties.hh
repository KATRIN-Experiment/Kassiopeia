#ifndef KFMCubicSpaceTreeProperties_HH__
#define KFMCubicSpaceTreeProperties_HH__

#include <string>

namespace KEMField
{

/**
*
*@file KFMCubicSpaceTreeProperties.hh
*@class KFMCubicSpaceTreeProperties
*@brief simple container for a small set of parameters that all nodes in a tree share
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
* J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM> class KFMCubicSpaceTreeProperties
{
  public:
    KFMCubicSpaceTreeProperties() : fCubicNeighborOrder(0), fMaxTreeDepth(0), fCurrentMaxUniqueID(0), fTreeID("")
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fDimSize[i] = 0;
            fTopLevelDimSize[i] = 0;
        };
    };

    virtual ~KFMCubicSpaceTreeProperties() = default;
    ;

  public:
    //unique id for the entire tree
    void SetTreeID(const std::string& tree_id)
    {
        fTreeID = tree_id;
    };
    std::string GetTreeID() const
    {
        return fTreeID;
    };

    //max depth of the tree
    void SetMaxTreeDepth(unsigned int depth)
    {
        fMaxTreeDepth = depth;
    };
    unsigned int GetMaxTreeDepth() const
    {
        return fMaxTreeDepth;
    };

    //nodes which are less than or equal this number of nodes away are considered neighbors
    void SetCubicNeighborOrder(unsigned int cn_order)
    {
        fCubicNeighborOrder = cn_order;
    };
    unsigned int GetCubicNeighborOrder() const
    {
        return fCubicNeighborOrder;
    };

    //get the number of dimension of the divisions of the tree
    unsigned int GetNDimensions() const
    {
        return NDIM;
    };

    //get/set the size of each dimension
    unsigned int GetDimension(unsigned int dim_index) const
    {
        return fDimSize[dim_index];
    }

    const unsigned int* GetDimensions() const
    {
        return fDimSize;
    };
    void GetDimensions(unsigned int* dim_size) const
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            dim_size[i] = fDimSize[i];
        }
    }

    void SetDimensions(const unsigned int* dim_size)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fDimSize[i] = dim_size[i];
        }
    }

    //get/set the size of each dimension for the top level of the tree
    unsigned int GetTopLevelDimension(unsigned int dim_index) const
    {
        return fTopLevelDimSize[dim_index];
    }

    const unsigned int* GetTopLevelDimensions() const
    {
        return fTopLevelDimSize;
    };
    void GetTopLevelDimensions(unsigned int* dim_size) const
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            dim_size[i] = fTopLevelDimSize[i];
        }
    }

    void SetTopLevelDimensions(const unsigned int* dim_size)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fTopLevelDimSize[i] = dim_size[i];
        }
    }

    int RegisterNode()
    {
        int id = fCurrentMaxUniqueID;
        fCurrentMaxUniqueID++;
        return id;
    }

    unsigned int GetNNodes() const
    {
        return fCurrentMaxUniqueID;
    };

  private:
    unsigned int fDimSize[NDIM];  //the number divisions in each dimension of the sub-division
    unsigned int fTopLevelDimSize[NDIM];
    unsigned int fCubicNeighborOrder;
    unsigned int fMaxTreeDepth;
    unsigned int fCurrentMaxUniqueID;

    std::string fTreeID;  //the id pertaining to the entire tree
};


}  // namespace KEMField


#endif /* KFMCubicSpaceTreeProperties_H__ */
