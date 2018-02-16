#ifndef KFMBasisData_HH__
#define KFMBasisData_HH__



namespace KEMField
{

/*
*
*@file KFMBasisData.hh
*@class KFMBasisData
*@brief wrapper for basis data
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug 29 09:40:36 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int BasisDimensionality>
class KFMBasisData
{
    public:
        KFMBasisData(){};

        KFMBasisData(const KFMBasisData& copyObject)
        {
            for(unsigned int i=0; i<BasisDimensionality; i++)
            {
                fData[i] = copyObject.fData[i];
            }
        }

        virtual ~KFMBasisData(){};

        unsigned int GetNDimensions() const {return BasisDimensionality;};

        inline KFMBasisData<BasisDimensionality>& operator= (const KFMBasisData<BasisDimensionality>& rhs)
        {
            if(&rhs != this)
            {
                for(unsigned int i=0; i<BasisDimensionality; i++)
                {
                    fData[i] = rhs.fData[i];
                }
            }
            return *this;
        }

        void Clear()
        {
            for(unsigned int i=0; i<BasisDimensionality; i++)
            {
                fData[i] = 0.;
            }
        }

        //access elements
        inline double& operator[](unsigned int i);
        inline const double& operator[](unsigned int i) const;


    private:

        double fData[BasisDimensionality];

};


template<unsigned int BasisDimensionality>
inline double& KFMBasisData<BasisDimensionality>::operator[](unsigned int i)
{
    return fData[i];
}

template<unsigned int BasisDimensionality>
inline const double& KFMBasisData<BasisDimensionality>::operator[](unsigned int i) const
{
    return fData[i];
}


}

#endif /* KFMBasisData_H__ */
