#ifndef KFMIdentityPair_HH__
#define KFMIdentityPair_HH__

namespace KEMField
{

/*
*
*@file KFMIdentityPair.hh
*@class KFMIdentityPair
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 14:50:24 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMIdentityPair
{
  public:
    KFMIdentityPair() : fID(0), fMappedID(0)
    {
        ;
    };
    KFMIdentityPair(unsigned int id, unsigned int mapped_id) : fID(id), fMappedID(mapped_id)
    {
        ;
    };
    virtual ~KFMIdentityPair() = default;
    ;

    unsigned int GetID() const
    {
        return fID;
    }

    void SetID(const unsigned int& id)
    {
        fID = id;
    }


    unsigned int GetMappedID() const
    {
        return fMappedID;
    }

    void SetMappedID(const unsigned int& id)
    {
        fMappedID = id;
    }


  private:
    unsigned int fID;
    unsigned int fMappedID;
};


}  // namespace KEMField


#endif /* KFMIdentityPair_H__ */
