#ifndef Kassiopeia_KSMathInterpolator_h_
#define Kassiopeia_KSMathInterpolator_h_

#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathSystem.h"

#include <cmath>
#include <list>
#include <utility>
#include <vector>

namespace Kassiopeia
{
template<class XType> class KSMathInterpolator;

template<class XValueType, class XDerivativeType, class XErrorType>
class KSMathInterpolator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>
{
  public:
    KSMathInterpolator();
    virtual ~KSMathInterpolator();

  public:
    virtual void
    Interpolate(double aTime,
                const KSMathIntegrator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& anIntegrator,
                const KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& aDifferentiator,
                const XValueType& anInitialValue, const XValueType& aFinalValue, const double& aStep,
                XValueType& anInterpolatedValue) const = 0;

    virtual double DistanceMetric(const XValueType& valueA, const XValueType& valueB) const;

    //use tolerance and max segment parameters to govern recursive subdivision the step into linear segments
    virtual void GetPiecewiseLinearApproximation(
        double aTolerance, unsigned int nMaxSegments,
        double
            anInitialTime, /*have to pass start time parameter because we cannot rely on in being part of the XValueType*/
        double aFinalTime, /*have to pass end time parameter because we cannot rely on in being part of the XValueType*/
        const KSMathIntegrator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& anIntegrator,
        const KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& aDifferentiator,
        const XValueType& anInitialValue, const XValueType& aFinalValue,
        std::vector<XValueType>* interpolatedValues) const;

    //use fixed number of segments to evenly sample the entire step
    virtual void GetFixedPiecewiseLinearApproximation(
        unsigned int nSegments,
        double
            anInitialTime, /*have to pass start time parameter because we cannot rely on in being part of the XValueType*/
        double aFinalTime, /*have to pass end time parameter because we cannot rely on in being part of the XValueType*/
        const KSMathIntegrator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& anIntegrator,
        const KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& aDifferentiator,
        const XValueType& anInitialValue, const XValueType& aFinalValue,
        std::vector<XValueType>* interpolatedValues) const;


    typedef typename std::pair<XValueType, bool> XValueFlagPair;
    typedef typename std::list<XValueFlagPair> XValueFlagPairList;
    typedef typename XValueFlagPairList::iterator XValueFlagPairListIter;
};

template<class XValueType, class XDerivativeType, class XErrorType>
KSMathInterpolator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::KSMathInterpolator()
{}

template<class XValueType, class XDerivativeType, class XErrorType>
KSMathInterpolator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::~KSMathInterpolator()
{}

template<class XValueType, class XDerivativeType, class XErrorType>
double KSMathInterpolator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::DistanceMetric(
    const XValueType& valueA, const XValueType& valueB) const
{
    //default is euclidean norm over all variabls present in the value type
    //this can be overloaded by the user depending on the value type
    //(e.g. may want only position variables to contribute and ignore momentum, etc)
    double dist = 0.0;
    double del;
    for (unsigned int i = 0; i < XValueType::sDimension; i++) {
        del = valueA[i] - valueB[i];
        dist += del * del;
    }
    return std::sqrt(dist);
}

template<class XValueType, class XDerivativeType, class XErrorType>
void KSMathInterpolator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::GetPiecewiseLinearApproximation(
    double aTolerance, unsigned int nMaxSegments,
    double
        anInitialTime, /*have to pass start time parameter because we cannot rely on in being part of the XValueType*/
    double aFinalTime, /*have to pass end time parameter because we cannot rely on in being part of the XValueType*/
    const KSMathIntegrator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& anIntegrator,
    const KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& aDifferentiator,
    const XValueType& anInitialValue, const XValueType& aFinalValue, std::vector<XValueType>* interpolatedValues) const
{
    interpolatedValues->clear();

    //list of <value, bisect-flag> pairs
    XValueFlagPairList tValues;
    tValues.push_back(XValueFlagPair(anInitialValue, true));
    tValues.push_back(XValueFlagPair(aFinalValue, false));
    auto tFirst = tValues.begin();
    auto tNext = tFirst;
    tNext++;

    //list of time parameters for each corresponding value
    std::list<double> tValueTimes;
    tValueTimes.push_back(anInitialTime);
    tValueTimes.push_back(aFinalTime);
    auto tFirstTime = tValueTimes.begin();
    auto tNextTime = tFirstTime;
    tNextTime++;

    //temp variables for work
    double tMidTime;
    double tTimeStep;
    XValueType tSegmentMidPoint;
    XValueType tCurveMidPoint;
    unsigned int segmentCount = 1;
    bool haveUncheckedSegments = true;

    //now we recursively bisect until we reach the appropriate tolerance or max number of segments
    while (haveUncheckedSegments && (segmentCount < nMaxSegments)) {
        tFirst = tValues.begin();
        tNext = tFirst;
        tNext++;
        tFirstTime = tValueTimes.begin();
        tNextTime = tFirstTime;
        tNextTime++;

        //assume we have check all the segments until we encounter one otherwise
        haveUncheckedSegments = false;

        //we do a breadth-first search for segments to bisect
        while ((tNext != tValues.end()) && (segmentCount < nMaxSegments)) {
            //check if need for bisection flag is true
            if (tFirst->second) {
                //compute line segment and curve mid-points
                tSegmentMidPoint = 0.5 * ((tFirst->first) + (tNext->first));
                tMidTime = 0.5 * (*tFirstTime + *tNextTime);
                tTimeStep = tMidTime - anInitialTime;
                Interpolate(anInitialTime,
                            anIntegrator,
                            aDifferentiator,
                            anInitialValue,
                            aFinalValue,
                            tTimeStep,
                            tCurveMidPoint);

                //distance exceeds tolerance?
                double dist = DistanceMetric(tSegmentMidPoint, tCurveMidPoint);
                if (dist > aTolerance) {
                    tFirst = tValues.insert(tNext, XValueFlagPair(tCurveMidPoint, true));
                    tFirst++;
                    tNext = tFirst;
                    tNext++;
                    tFirstTime = tValueTimes.insert(tNextTime, tMidTime);
                    tFirstTime++;
                    tNextTime = tFirstTime;
                    tNextTime++;
                    haveUncheckedSegments = true;  //will have to inspect this new segment on next pass
                    segmentCount++;
                }
                else {
                    //no longer need to consider this segment for bisection
                    tFirst->second = false;
                    tFirst++;
                    tNext++;
                    tFirstTime++;
                    tNextTime++;
                }
            }
            else {
                tFirst++;
                tNext++;
                tFirstTime++;
                tNextTime++;
            }
        }
    }

    //now fill up the output std::vector with the intermediate state results (in time order)
    for (tFirst = tValues.begin(); tFirst != tValues.end(); tFirst++) {
        interpolatedValues->push_back(tFirst->first);
    }
}


template<class XValueType, class XDerivativeType, class XErrorType>
void KSMathInterpolator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::GetFixedPiecewiseLinearApproximation(
    unsigned int nSegments,
    double
        anInitialTime, /*have to pass start time parameter because we cannot rely on in being part of the XValueType*/
    double aFinalTime, /*have to pass end time parameter because we cannot rely on in being part of the XValueType*/
    const KSMathIntegrator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& anIntegrator,
    const KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>& aDifferentiator,
    const XValueType& anInitialValue, const XValueType& aFinalValue, std::vector<XValueType>* interpolatedValues) const
{
    interpolatedValues->clear();
    double delta = (aFinalTime - anInitialTime) / ((double) nSegments);

    //insert the first state
    interpolatedValues->push_back(anInitialValue);

    //compute linear sample of intermediate states
    XValueType tState;
    for (unsigned int i = 1; i < nSegments; i++) {
        double tTimeStep = i * delta;
        Interpolate(anInitialTime, anIntegrator, aDifferentiator, anInitialValue, aFinalValue, tTimeStep, tState);
        interpolatedValues->push_back(tState);
    }

    //insert the last state
    interpolatedValues->push_back(aFinalValue);
}

}  // namespace Kassiopeia

#endif
