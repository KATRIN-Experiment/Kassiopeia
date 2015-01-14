#ifndef KVMFixedArray_H
#define KVMFixedArray_H

#include <cstddef>
#include <vector>

/**
*
*@file KVMFixedArray.hh
*@class KVMFixedArray
*@brief Class intended to mimic a super minimal sub-set of the functionality of
* std::array in the C++0X standard, this only intended to be used
* to fix the interface to classes like KVMCompactCurve or KVMCompactSurface.
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul  7 11:23:06 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename T, unsigned int ArrSize>
class KVMFixedArray
{
    public:
        KVMFixedArray(){array = new T[ArrSize];};
        virtual ~KVMFixedArray(){delete[] array;};

        inline KVMFixedArray(const KVMFixedArray& copyObject)
        {
            array = new T[ArrSize];
            for(unsigned int i=0; i<ArrSize; i++)
            {
                array[i] = copyObject.array[i];
            }
        }

        KVMFixedArray& operator=(const KVMFixedArray& rhs)
        {
            if(this != &rhs)
            {
                for(unsigned int i =0; i<ArrSize; i++)
                {
                    array[i] = rhs.array[i];
                }
            }
            return *this;
        }

        T& operator[](unsigned int i)
        {
            return array[i];
        }

        const T& operator[](unsigned int i) const
        {
            return array[i];
        }

        unsigned int Size(){return ArrSize;};

        const T* GetBareArray() const {return array;};
        T* GetBareArray() {return array;};

    private:

    T* array;

};


#endif /* KVMFixedArray_H */
