#ifndef KGSpaceTreeProperties_HH__
#define KGSpaceTreeProperties_HH__

#include <string>

namespace KGeoBag
{

/**
*
*@file KGSpaceTreeProperties.hh
*@class KGSpaceTreeProperties
*@brief simple container for a small set of parameters that all nodes in a tree share
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
* J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM> class KGSpaceTreeProperties
{
  public:
    KGSpaceTreeProperties() : fMaxTreeDepth(0), fCurrentMaxUniqueID(0), fTreeID("")
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fDimSize[i] = 0;
        };
    };

    virtual ~KGSpaceTreeProperties(){};

  public:
    //unique id for the entire tree
    void SetTreeID(std::string tree_id)
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
    void SetNeighborOrder(unsigned int cn_order)
    {
        fNeighborOrder = cn_order;
    };
    unsigned int GetNeighborOrder() const
    {
        return fNeighborOrder;
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
    unsigned int fNeighborOrder;
    unsigned int fMaxTreeDepth;
    unsigned int fCurrentMaxUniqueID;
    std::string fTreeID;  //the id pertaining to the entire tree
};


}  // namespace KGeoBag


#endif /* KGSpaceTreeProperties_H__ */
