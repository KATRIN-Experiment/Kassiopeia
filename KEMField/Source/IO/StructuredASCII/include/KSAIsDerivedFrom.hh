#ifndef KSAIsDerivedFrom_HH__
#define KSAIsDerivedFrom_HH__


/**
*
*@file KSAIsDerivedFrom.hh
*@class KSAIsDerivedFrom
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Feb  4 13:46:14 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//This is some crazy Alexandrescu stuff I won't pretend to understand that helps
//us make a template specialization when a particular template parameter is
//derived from a particular type

//When D is derived from B, then KSAIsDerivedFrom::Is = 1,
//When D is not derived from B, the KSAIsDerivedFrom::Is = 0

//taken from: http://www.gotw.ca/publications/mxc++-item-4.htm


template<typename D, typename B> class KSAIsDerivedFrom
{
    class No
    {};
    class Yes
    {
        No no[3];
    };

    static Yes Test(B*);  // not defined
    static No Test(...);  // not defined

    static void Constraints(D* p)
    {
        B* pb = p;
        pb = p;
    }

  public:
    enum
    {
        Is = sizeof(Test(static_cast<D*>(nullptr))) == sizeof(Yes)
    };

    KSAIsDerivedFrom()
    {
        void (*p)(D*) = Constraints;
    }
};


#endif /* KSAIsDerivedFrom_H__ */
