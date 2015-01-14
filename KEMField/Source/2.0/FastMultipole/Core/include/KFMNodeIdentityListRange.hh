#ifndef KFMNodeIdentityListRange_HH__
#define KFMNodeIdentityListRange_HH__


namespace KEMField
{

/*
*
*@file KFMNodeIdentityListRange.hh
*@class KFMNodeIdentityListRange
*@brief
* Assuming a corresponding appropriately reordered list of identity indices this class specifies
* the range of that list which a particular node is responsible for
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jun 14 13:34:10 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMNodeIdentityListRange
{
    public:
        KFMNodeIdentityListRange():fStartIndex(0),fLength(0){};
        virtual ~KFMNodeIdentityListRange(){};

        void SetStartIndex(unsigned int i) {fStartIndex = i;};
        void SetLength(unsigned int i) {fLength = i;};

        unsigned int GetStartIndex() const {return fStartIndex;};
        unsigned int GetStopIndex() const {return fStartIndex + fLength;};
        unsigned int GetLength() const {return fLength;};

    private:

        unsigned int fStartIndex;
        unsigned int fLength;

};



}


#endif /* KFMNodeIdentityListRange_H__ */
