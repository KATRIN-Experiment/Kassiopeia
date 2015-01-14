#ifndef KFMNamedScalarDataCollection_HH__
#define KFMNamedScalarDataCollection_HH__

#include <string>
#include <vector>

#include "KFMNamedScalarData.hh"

#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField
{

/*
*
*@file KFMNamedScalarDataCollection.hh
*@class KFMNamedScalarDataCollection
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr 23 10:38:33 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMNamedScalarDataCollection: public KSAInputOutputObject
{
    public:
        KFMNamedScalarDataCollection(){};
        virtual ~KFMNamedScalarDataCollection(){};

        const KFMNamedScalarData* GetDataWithName(std::string name) const;
        KFMNamedScalarData* GetDataWithName(std::string name);
        void AddData(const KFMNamedScalarData& data);

        unsigned int GetNDataSets() const {return fData.size();};
        const KFMNamedScalarData* GetDataSetWithIndex(unsigned int i) const {return &(fData[i]);};
        KFMNamedScalarData* GetDataSetWithIndex(unsigned int i) {return &(fData[i]);};

        virtual void DefineOutputNode(KSAOutputNode* node) const;
        virtual void DefineInputNode(KSAInputNode* node);
        virtual const char* ClassName() const { return "KFMNamedScalarDataCollection"; };


    private:

        std::vector< KFMNamedScalarData > fData;


};


DefineKSAClassName( KFMNamedScalarDataCollection );

}


#endif /* KFMNamedScalarDataCollection_H__ */
