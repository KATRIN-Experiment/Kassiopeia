#ifndef KMATHLAGRANGEINTERPOLATOR_H_
#define KMATHLAGRANGEINTERPOLATOR_H_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <vector>

//lagrange interpolator from ferenc
// n: number of points for interpolation

namespace katrin {
	class KMathLagrangeInterpolator
	{
	public:
		KMathLagrangeInterpolator(size_t n = 4) : fN(n) { }

		void AddValue(double point, double value);
		size_t Size() const { return fData.size(); }
		bool Empty() const { return fData.empty(); }
		void Reset();
		double Calculate(double point) const;
		double GetMinX() const;
		double GetMaxX() const;

	private:
		size_t fN;
		std::map<double, double> fData;
	};

	inline void KMathLagrangeInterpolator::AddValue(double point, double value)
	{
		fData.insert( std::pair<double,double>(point, value));
	}

	inline void KMathLagrangeInterpolator::Reset()
	{
		fData.clear();
	}

	inline double KMathLagrangeInterpolator::GetMinX() const
	{
		return (fData.empty())
		        ? std::numeric_limits<double>::quiet_NaN()
                : fData.cbegin()->first;
	}

	inline double KMathLagrangeInterpolator::GetMaxX() const
	{
        return (fData.empty())
                ? std::numeric_limits<double>::quiet_NaN()
                : fData.crbegin()->first;
	}

	inline double KMathLagrangeInterpolator::Calculate( double point) const
	{
	    const size_t n = std::min(fN, fData.size());

		//first find 4 points close to point
		std::map<double,double> tClosePoints;
		auto tItLower = fData.lower_bound( point );
		auto tItUpper = tItLower;

		size_t tCount = 0;
		while ( true )
		{
			if ( tItLower != fData.begin() )
			{
				tItLower--;
				tCount++;
			}
			if ( tCount == n ) break;
			if ( tItUpper != fData.end() )
			{
				tItUpper++;
				tCount++;
			}
			if ( tCount == n ) break;
		}

		tClosePoints.insert( tItLower, tItUpper);

		//iam sorry for the mess starting from here
		double tValue = 0;
		const size_t nClosePoints = tClosePoints.size();
		std::vector<double> A(nClosePoints), B(nClosePoints);

		size_t tIndex1 = 0;
		for ( auto tIt1 = tClosePoints.cbegin(); tIt1 != tClosePoints.cend(); tIt1++ )
		{
			size_t tIndex2 = 0;
			for ( auto tIt2 = tClosePoints.cbegin(); tIt2 != tClosePoints.cend(); tIt2++ )
			{
				A[tIndex2] = ( point - tIt2->first );
				B[tIndex2] = ( tIt1->first - tIt2->first );
				tIndex2++;
			}
			A[tIndex1] = 1.0;
			B[tIndex1] = 1.0;
			double AA = 1.0, BB = 1.0;
			tIndex2 = 0;
			for ( size_t i = 0; i < nClosePoints; ++i )
			{
				AA = AA * A[tIndex2];
				BB = BB * B[tIndex2];
				tIndex2++;
			}
			tValue += tIt1->second * AA / BB;
			tIndex1++;
		}

		return tValue;
	}


}




#endif /* KMATHLAGRANGEINTERPOLATOR_H_ */
