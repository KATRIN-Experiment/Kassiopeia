/*
 * KMathBilinearInterpolator.h
 *
 *  Created on: 15.01.2014
 *      Author: oertlin
 */

#ifndef KMATHBILINEARINTERPOLATOR_H_
#define KMATHBILINEARINTERPOLATOR_H_

#include <map>
#include <vector>
#include <assert.h>
using namespace std;

namespace katrin {
	/**
	 * Bilinear interpolation for regular grids as described in Wikipedia:
	 * http://en.wikipedia.org/wiki/Bilinear_interpolation
	 */
	template<class T = double>
	class KMathBilinearInterpolator {
		typedef std::map<T, std::map<T, T> > PointMap;
	public:
		KMathBilinearInterpolator() {};
		~KMathBilinearInterpolator() {};

		/**
		 * \brief Adds a point. Note, that a regular grid is needed.
		 *
		 * \param x
		 * \param y
		 * \param value The value at (x, y)
		 */
		void AddPoint(const T x, const T y, const T value);

		/**
		 * \brief Adds a point. Note, that a regular grid is needed.
		 *
		 * \param point
		 * \param value The value at (x, y)
		 */
		void AddPoint(const std::vector<T> &point, const T value) {assert(point.size() == 2); AddPoint(point[0], point[1], value);}

		/**
		 * \brief Adds a point. Note, that a regular grid is needed.
		 *
		 * \param point
		 * \param value The value at (x, y)
		 */
		void AddPoint(const T *point, const T value) {assert(point != 0); AddPoint(point[0], point[1], value);};

		/**
		 * \brief Calculates the value for the given x and y.
		 *
		 * \param x
		 * \param y
		 * \return The interpolated value
		 */
		T GetValue(const T &x, const T &y) const;

		/**
		 * \brief Calculates the value for the given x and y. Note, that "point" have to
		 * hold at least 2 elements.
		 *
		 * \param x
		 * \param y
		 * \return The interpolated value
		 */
		T GetValue(const std::vector<T> &point) const {assert(point.size() == 2); return GetValue(point[0], point[1]);};

		/**
		 * \brief Calculates the value for the given x and y. Note, that "point" have to
		 * hold at least 2 elements.
		 *
		 * \param x
		 * \param y
		 * \return The interpolated value
		 */
		T GetValue(const T *point) const {assert(point != 0); return GetValue(point[0], point[1]);};

		/**
		 * \brief Removes all added points.
		 */
		void Reset() {fPoints.clear();}

		PointMap* GetPoints() {return &fPoints;}

	private:
		PointMap fPoints;

	private:
		T GetLinearInterpolation(const T &x, const T &x1, const T &y1, const T &x2, const T &y2) const;
	};
}

template<class T>
void katrin::KMathBilinearInterpolator<T>::AddPoint(const T x, const T y, const T value) {
	typename PointMap::iterator xIt = fPoints.find(x);

	if(xIt == fPoints.end()) {
		// x does not exists
		xIt = fPoints.insert(std::pair<T, std::map<T, T> >(x, std::map<T, T>())).first;
	}

	// Looking for the y...
	typename std::map<T, T>::iterator yIt = xIt->second.find(y);

	if(yIt != xIt->second.end()) {
		// if this point does exists...
		xIt->second[y] = value;
	} else {
		xIt->second.insert(std::pair<T, T>(y, value));
	}
}

template<class T>
T katrin::KMathBilinearInterpolator<T>::GetValue(const T &x, const T &y) const {
	T x1, x2, y1, y2, value;

	typename PointMap::const_iterator xUpper = fPoints.lower_bound(x);

	assert(fPoints.begin()->first <= x);
	assert(xUpper != fPoints.end());

	typename std::map<T, T>::const_iterator yUpper = xUpper->second.lower_bound(y);
	typename std::map<T, T>::const_iterator yLower;

    T assert_ymin = xUpper->second.begin()->first;
    (void)assert_ymin;

    assert(assert_ymin <= y);
	assert(yUpper != xUpper->second.end());

//    std::cout << "Number of Points " << fPoints.size() << std::endl;
	// Test, if x is equal to the point
//    std::cout << "x: " << x << std::endl;

//    std::cout << "xUpper->first: " << xUpper->first << std::endl;
//    std::cout << "yUpper->first: " << yUpper->first << std::endl;
	if(xUpper->first == x) {
//        std::cout << " FIRST IF " << std::endl;
		// Now test, if y also matches a point
		if(yUpper->first == y) {
//            std::cout << " IF ! " << std::endl;
            // Okay, nothing to calculate...
			return fPoints.at(x).at(y);
		} else {
			// Okay, now we have to use a simple 1dim interpolation
//            std::cout << " ELSE ! " << std::endl;
			yLower = yUpper;
			--yLower;
			return GetLinearInterpolation(y, yLower->first, yLower->second, yUpper->first, yUpper->second);
		}
	}

	typename PointMap::const_iterator xLower = xUpper;
	--xLower;

	x1 = xLower->first;
	x2 = xUpper->first;

	// Test, if y is equal to the point
	if(yUpper->first == y) {
		// Okay, now we have to use a simple 1dim interpolation
		return GetLinearInterpolation(x, xLower->first, xLower->second.at(y), xUpper->first, xUpper->second.at(y));
	}

	yLower = yUpper;
	--yLower;
	y1 = yLower->first;
	y2 = yUpper->first;

	// Calculate value
	value = fPoints.at(x1).at(y1) * (x2 - x) * (y2 - y) +
			fPoints.at(x2).at(y1) * (x - x1) * (y2 - y) +
			fPoints.at(x1).at(y2) * (x2 - x) * (y - y1) +
			fPoints.at(x2).at(y2) * (x - x1) * (y - y1);
	value /= (x2 - x1) * (y2 - y1);

	return value;
}

template<class T>
T katrin::KMathBilinearInterpolator<T>::GetLinearInterpolation(const T &x, const T &x1, const T &y1, const T &x2, const T &y2) const {
	T slope = (y2 - y1) / (x2 - x1);
	T y0 = y1 - slope * x1;

	return slope * x + y0;
}

#endif /* KMATHBILINEARINTERPOLATOR_H_ */
