#ifndef KSCyclicIterator_h
#define KSCyclicIterator_h

/**
 @file
 @brief contains KSCyclicIterator
 @details
 <b>Revision History:</b>
 \verbatim
 Date         Name        Brief description
 -----------------------------------------------
 02.07.2010  D. Furse   First version
 \endverbatim
*/

/*!
 @class Kassiopeia::KSCyclicIterator
 @author D. Furse

 @brief the cyclic iterator iterates over a bunch of allocated memory

 @details
 <b>Detailed Description:</b><br>
	It can be used in situations where you have to keep track of the past N results to try to predict the N+1 result, e.g. the Predictor Corrector Solver.
	Instead of copying each result from position i to i-1 and store the latest result always in the same position N,
	just N-1 pointers to the last results get shifted and the newest result is stored in the place of the oldest and outdated result.
	this is significantly faster than copying the objects themselves...

*/


#include "Rtypes.h"
#include "TMath.h"

namespace Kassiopeia
{
template<class XType> class KSCyclicIterator
{
  public:
    KSCyclicIterator();
    KSCyclicIterator(const KSCyclicIterator<XType>& source);  //copy constructor
    virtual ~KSCyclicIterator();

    void SetArray(XType* const block, const UInt_t size);

    XType& operator*();  //dereferences the iterator

    void ToMark();  // moves the current iterator position to the marked position
    void IncrementMark(const Int_t pos);
    void IncrementPosition(const Int_t pos);

    void operator+();  //cyclically increments the current iterator position and the marked position
    void operator-();  //cyclically decrements the current iterator position and the marked position
    Bool_t
    operator++();  // cyclically increments the current iterator position, returning false if it hits the marked position
    Bool_t
    operator--();  // cyclically decrements the current iterator position, returning false if it hits the marked position

  private:
    XType* fBlock;
    UInt_t fSize;
    UInt_t fMarkedIndex;
    UInt_t fCurrentIndex;
};

template<class XType>
inline KSCyclicIterator<XType>::KSCyclicIterator() : fBlock(0), fSize(0), fMarkedIndex(0), fCurrentIndex(0)
{}
template<class XType>
inline KSCyclicIterator<XType>::KSCyclicIterator(const KSCyclicIterator<XType>& source) :
    fBlock(source.fBlock),
    fSize(source.fSize),
    fMarkedIndex(source.fMarkedIndex),
    fCurrentIndex(source.fCurrentIndex)
{}
template<class XType> inline KSCyclicIterator<XType>::~KSCyclicIterator() = default;

template<class XType> inline void KSCyclicIterator<XType>::SetArray(XType* const block, const UInt_t size)
{
    fBlock = block;
    fSize = size;
    fMarkedIndex = 0;
    fCurrentIndex = 0;
    return;
}

template<class XType> inline XType& KSCyclicIterator<XType>::operator*()
{
    return fBlock[fCurrentIndex];
}

template<class XType> inline void KSCyclicIterator<XType>::ToMark()
{
    fCurrentIndex = fMarkedIndex;
    return;
}
template<class XType> inline void KSCyclicIterator<XType>::IncrementMark(const Int_t pos)
{
    IncrementPosition(pos);
    Int_t NewPos = ((Int_t)(fMarkedIndex + pos)) % ((Int_t)(fSize));
    if (NewPos < 0) {
        NewPos = NewPos + fSize;
    }
    fMarkedIndex = NewPos;
    return;
}
template<class XType> inline void KSCyclicIterator<XType>::IncrementPosition(const Int_t pos)
{
    Int_t NewPos = ((Int_t)(fCurrentIndex + pos)) % ((Int_t)(fSize));
    if (NewPos < 0) {
        NewPos = NewPos + fSize;
    }
    fCurrentIndex = NewPos;
    return;
}
template<class XType> void KSCyclicIterator<XType>::operator+()
{
    fMarkedIndex++;
    if (fMarkedIndex == fSize) {
        fMarkedIndex = 0;
    }
    fCurrentIndex++;
    if (fCurrentIndex == fSize) {
        fCurrentIndex = 0;
    }
    return;
}
template<class XType> void KSCyclicIterator<XType>::operator-()
{
    if (fMarkedIndex == 0) {
        fMarkedIndex = fSize;
    }
    fMarkedIndex--;

    if (fCurrentIndex == 0) {
        fCurrentIndex = fSize;
    }
    fCurrentIndex--;
    return;
}
template<class XType> Bool_t KSCyclicIterator<XType>::operator++()
{
    fCurrentIndex++;
    if (fCurrentIndex == fSize) {
        fCurrentIndex = 0;
    }
    if (fCurrentIndex == fMarkedIndex) {
        return false;
    }
    return true;
}
template<class XType> Bool_t KSCyclicIterator<XType>::operator--()
{
    if (fCurrentIndex == 0) {
        fCurrentIndex = fSize;
    }
    fCurrentIndex--;
    if (fCurrentIndex == fMarkedIndex) {
        return false;
    }
    return true;
}

}  // end namespace Kassiopeia

#endif  // end ifndef KSCyclicIterator_h
