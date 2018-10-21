#ifndef KFMElectrostaticBasisDataExtractor_HH__
#define KFMElectrostaticBasisDataExtractor_HH__

#include "KBasis.hh"
#include "KElectrostaticBasis.hh"
#include "KSurfaceTypes.hh"
#include "KSurfaceVisitors.hh"

#include "KFMBasisData.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticBasisDataExtractor.hh
*@class KFMElectrostaticBasisDataExtractor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Sep  1 13:11:42 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMElectrostaticBasisDataExtractor: public KSelectiveVisitor<KBasisVisitor,KTYPELIST_1(KElectrostaticBasis)>
{
    public:
        KFMElectrostaticBasisDataExtractor(){};
        virtual ~KFMElectrostaticBasisDataExtractor(){};

        using KSelectiveVisitor<KBasisVisitor, KTYPELIST_1(KElectrostaticBasis)>::Visit;

        void Visit(KElectrostaticBasis& basis)
        {
            fCurrentBasisData[0] = basis.GetSolution(0);
        }

        KFMBasisData<1> GetBasisData() const {return fCurrentBasisData;};

    private:

        KFMBasisData<1> fCurrentBasisData;
};




}

#endif /* KFMElectrostaticBasisDataExtractor_H__ */
