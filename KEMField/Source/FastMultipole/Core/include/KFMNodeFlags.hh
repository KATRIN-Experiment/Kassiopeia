#ifndef __KFMNodeFlags_H__
#define __KFMNodeFlags_H__

namespace KEMField
{

/**
*
*@file KFMNodeFlags.hh
*@class KFMNodeFlags
*@brief collection of externally managed character flags that apply to certain nodes
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Jul  6 15:33:50 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NFLAGS>
class KFMNodeFlags
{
    public:
        KFMNodeFlags()
        {
            for(unsigned int i=0; i<NFLAGS; i++){fFlags[i] = 0;};
        };

        virtual ~KFMNodeFlags(){};

        void SetFlag(unsigned int flag_index, char flag_value)
        {
            fFlags[flag_index] = flag_value;
        }

        char GetFlag(unsigned int flag_index) const
        {
            return fFlags[flag_index];
        }

    protected:
        /* data */

        char fFlags[NFLAGS];

};

}

#endif /* __KFMNodeFlags_H__ */
