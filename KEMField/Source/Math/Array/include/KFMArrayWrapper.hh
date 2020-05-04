#ifndef KFMArrayWrapper_HH__
#define KFMArrayWrapper_HH__

#include "KFMArrayMath.hh"

namespace KEMField
{

/*
*
*@file KFMArrayWrapper.hh
*@class KFMArrayWrapper
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Aug 24 12:52:33 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ArrayType, unsigned int NDIM> class KFMArrayWrapper
{
  public:
    KFMArrayWrapper()
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fBases[i] = 0;
            fDimensions[i] = 0;
        }
        fTotalArraySize = 0;
    }

    KFMArrayWrapper(ArrayType* data, const unsigned int* dim)
    {
        fData = data;

        for (unsigned int i = 0; i < NDIM; i++) {
            fBases[i] = 0;
            fDimensions[i] = dim[i];
        }
        fTotalArraySize = KFMArrayMath::TotalArraySize<NDIM>(fDimensions);
    }

    virtual ~KFMArrayWrapper()
    {
        ;
    };

    void SetData(ArrayType* ptr)
    {
        fData = ptr;
    }
    ArrayType* GetData()
    {
        return fData;
    };

    unsigned int GetArraySize() const
    {
        return KFMArrayMath::TotalArraySize<NDIM>(fDimensions);
    };

    void SetArrayDimensions(const unsigned int* array_dim)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fDimensions[i] = array_dim[i];
        }

        fTotalArraySize = KFMArrayMath::TotalArraySize<NDIM>(fDimensions);
    }

    void GetArrayDimensions(unsigned int* array_dim) const
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            array_dim[i] = fDimensions[i];
        }
    }

    const unsigned int* GetArrayDimensions() const
    {
        return fDimensions;
    }

    unsigned int GetArrayDimension(unsigned int dim_index) const
    {
        return fDimensions[dim_index];
    }

    void SetArrayBases(const int* array_bases)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fBases[i] = array_bases[i];
        }
    }

    void GetArrayBases(int* array_bases) const
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            array_bases[i] = fBases[i];
        }
    }


    unsigned int GetOffsetForIndices(const int* index)
    {
        unsigned int index_proxy[NDIM];
        for (unsigned int i = 0; i < NDIM; i++) {
            index_proxy[i] = (index[i] - fBases[i]);
        }
        return KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDimensions, index_proxy);
    }


    ArrayType& operator[](const int* index);

    const ArrayType& operator[](const int* index) const;

    ArrayType& operator[](unsigned int index);

    const ArrayType& operator[](unsigned int index) const;


  private:
    ArrayType* fData;  //raw pointer to multidimensional array
    unsigned int fDimensions[NDIM];
    int fBases[NDIM];

    unsigned int fTotalArraySize;
};

template<typename ArrayType, unsigned int NDIM>
ArrayType& KFMArrayWrapper<ArrayType, NDIM>::operator[](const int* index)
{
    unsigned int index_proxy[NDIM];
    for (unsigned int i = 0; i < NDIM; i++) {
        index_proxy[i] = index[i] - fBases[i];
    }

    //    #ifdef KFMArrayWrapper_DEBUG
    //        unsigned int offset = KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDimensions, index_proxy);
    //        if( offset >= fTotalArraySize)
    //        {
    //            kfmout<<"KFMArrayWrapper[]: Warning index out of range!: "<<offset<<" > "<<fTotalArraySize<<kfmendl;
    //            for(unsigned int i=0; i<NDIM; i++)
    //            {
    //                kfmout<<"index_proxy["<<i<<"] = "<<index_proxy[i]<<kfmendl;
    //                kfmout<<"index["<<i<<"] = "<<index[i]<<kfmendl;
    //                kfmout<<"base["<<i<<"] = "<<fBases[i]<<kfmendl;
    //                kfmout<<"dimension size["<<i<<"] = "<<fDimensions[i]<<kfmendl;
    //            }
    //            kfmexit(1);
    //        }
    //    #endif

    return fData[KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDimensions, index_proxy)];
}


template<typename ArrayType, unsigned int NDIM>
const ArrayType& KFMArrayWrapper<ArrayType, NDIM>::operator[](const int* index) const
{
    unsigned int index_proxy[NDIM];
    for (unsigned int i = 0; i < NDIM; i++) {
        index_proxy[i] = index[i] - fBases[i];
    }
    return fData[KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDimensions, index_proxy)];
}

template<typename ArrayType, unsigned int NDIM>
ArrayType& KFMArrayWrapper<ArrayType, NDIM>::operator[](unsigned int index)
{
    return fData[index];
}

template<typename ArrayType, unsigned int NDIM>
const ArrayType& KFMArrayWrapper<ArrayType, NDIM>::operator[](unsigned int index) const
{
    return fData[index];
}


}  // namespace KEMField


#endif /* KFMArrayWrapper_H__ */
