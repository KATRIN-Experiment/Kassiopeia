/**
 * @file KHistogram.h
 *
 * @date 03.12.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KOMMON_CORE_UTILITY_KHISTOGRAM_H_
#define KOMMON_CORE_UTILITY_KHISTOGRAM_H_

#include "KException.h"

#include <vector>
#include <utility>
#include <cmath>
#include <type_traits>
#include <ostream>

namespace katrin {

template<class XValueT = double, class XCountingT = double>
class KHistogram
{
public:
    typedef std::vector<XCountingT> data_type;

public:
    KHistogram(XValueT min = 0, XValueT max = 0, XValueT binWidth = 0);
    ~KHistogram();

    void Reset();

    XValueT GetMinValue() const { return fMinValue; }
    void SetMinValue(XValueT min);

    XValueT GetMaxValue() const { return fMaxValue; }
    void SetMaxValue(XValueT max);

    size_t GetNumberOfBins() const { return fNumberOfBins; }
    void SetNumberOfBins(size_t n);

    XValueT GetBinWidth() const;
    void SetBinWidth(XValueT width);

    void Fill(XValueT value, XCountingT amount = 1);

    size_t Size() const { return fData.size(); }
    XValueT BinCenter(size_t index) const { return fMinValue + GetBinWidth() * ((XValueT) index + 0.5); }
    XValueT BinLower(size_t index) const { return fMinValue + GetBinWidth() * ((XValueT) index); }
    XValueT BinUpper(size_t index) const { return fMinValue + GetBinWidth() * ((XValueT) index + 1.0); }
    const XCountingT& BinContent(size_t index) const { return fData[index]; }

    const data_type& Data() const { return fData; }
    const XCountingT& UnderFlow() const { return fUnderFlow; }
    const XCountingT& OverFlow() const { return fOverFlow; }

    void Trim(size_t nPadBins = 0);

    XCountingT Sum(bool includeUnderOverFlow = true) const;
    XCountingT Integral(bool includeUnderOverFlow = true) const { return Sum(includeUnderOverFlow) * GetBinWidth(); }
    void Scale(XCountingT factor);
    void Normalize(bool includeUnderOverFlow = true);

protected:
    void Initialize();

    XValueT fMinValue;
    XValueT fMaxValue;
    size_t fNumberOfBins;

    data_type fData;
    XCountingT fUnderFlow;
    XCountingT fOverFlow;
};

template<class XValueT, class XCountingT>
inline KHistogram<XValueT, XCountingT>::KHistogram(XValueT min, XValueT max, XValueT binWidth) :
    fMinValue(min),
    fMaxValue(max),
    fNumberOfBins(0)
{
    SetBinWidth(binWidth);
}

template<class XValueT, class XCountingT>
inline KHistogram<XValueT, XCountingT>::~KHistogram()
{ }

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::Reset()
{
    fData.assign( fNumberOfBins, 0 );
    fUnderFlow = 0;
    fOverFlow = 0;
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::SetMinValue(XValueT min)
{
    if (fMinValue == min)
        return;
    fMinValue = min;
    Reset();
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::SetMaxValue(XValueT max)
{
    if (fMaxValue == max)
        return;
    fMaxValue = max;
    Reset();
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::SetNumberOfBins(size_t n)
{
    if (fNumberOfBins == n)
        return;
    fNumberOfBins = n;
    Reset();
}

template<class XValueT, class XCountingT>
inline XValueT KHistogram<XValueT, XCountingT>::GetBinWidth() const
{
    return (fMaxValue - fMinValue) / (XValueT) fNumberOfBins;
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::SetBinWidth(XValueT width)
{
    size_t n = (width <= 0) ? 0 : lround( (fMaxValue - fMinValue) / width );
    if (n == 0) {
        n = 1;
        width = fMaxValue - fMinValue;
    }
    SetNumberOfBins(n);
    SetMaxValue(fMinValue + (XValueT) n * width);
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::Fill(XValueT value, XCountingT amount)
{
    if (value < fMinValue) {
        fUnderFlow += amount;
        return;
    }

    if (value >= fMaxValue) {
        fOverFlow += amount;
        return;
    }

    const size_t binNumber = (size_t) ((value - fMinValue) / GetBinWidth());
    fData[binNumber] += amount;
}

template<class XValueT, class XCountingT>
inline XCountingT KHistogram<XValueT, XCountingT>::Sum(bool includeUnderOverFlow) const
{
    XCountingT result = 0;
    for (const auto& p : fData)
        result += p;

    if (includeUnderOverFlow) {
        result += fUnderFlow;
        result += fOverFlow;
    }

    return result;
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::Scale(XCountingT factor)
{
    for (auto& p : fData)
        p *= factor;
    fUnderFlow *= factor;
    fOverFlow *= factor;
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::Normalize(bool includeUnderOverFlow)
{
    if (!std::is_floating_point<XCountingT>::value)
        throw KException() << "The function KHistogram<XValueT, XCountingT>::Normalize is not supported for integral XCountingT types.";

    const XCountingT integral = Integral(includeUnderOverFlow);
    if (integral == 0.0)
        return;

    Scale( 1.0 / integral );
}

template<class XValueT, class XCountingT>
inline void KHistogram<XValueT, XCountingT>::Trim(size_t nPadBins)
{
    size_t newStart = 0;
    size_t newEnd = fData.size();

    for (size_t i = 0; i < fData.size(); ++i) {
        if (fData[i] != 0) {
            newStart = i;
            newStart -= std::min(newStart, nPadBins);
            break;
        }
    }

    for (size_t i = fData.size(); i > 0; --i) {
        if (fData[i-1] != 0) {
            newEnd = i;
            newEnd += std::min(fData.size()-newEnd-1, nPadBins);
            break;
        }
    }

    const XValueT newMinValue = BinLower(newStart);
    const XValueT newMaxValue = BinLower(newEnd);

    fMinValue = newMinValue;
    fMaxValue = newMaxValue;

    fNumberOfBins = newEnd - newStart;

    fData.erase( fData.begin()+newEnd, fData.end() );
    fData.erase( fData.begin(), fData.begin()+newStart);
}

template<class XValueT, class XCountingT>
inline std::ostream& operator<< (std::ostream& o, const KHistogram<XValueT, XCountingT>& h)
{
    for (size_t i = 0; i < h.Size(); ++i)
        o << h.BinCenter(i) << "\t" << h.BinContent(i) << "\n";
    return o;
}

using KHistogramD = KHistogram<double, double>;
using KHistogramI = KHistogram<double, uint32_t>;

}

#endif /* KOMMON_CORE_UTILITY_KHISTOGRAM_H_ */
