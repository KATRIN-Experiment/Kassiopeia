/*
 * KEMStreamableThreeVector.hh
 *
 *  Created on: 13.05.2015
 *      Author: gosda
 */

#ifndef KEMSTREAMABLETHREEVECTOR_HH_
#define KEMSTREAMABLETHREEVECTOR_HH_

#include <iostream>
#include "KThreeVector_KEMField.hh"

namespace KEMField {
class KEMStreamableThreeVector {
public:
	KEMStreamableThreeVector() {
		fData[0] = 0;
		fData[1] = 0;
		fData[2] = 0;
	}
	explicit KEMStreamableThreeVector(const KThreeVector& vec) {
		fData[0] = vec.X();
		fData[1] = vec.Y();
		fData[2] = vec.Z();
	}

	KThreeVector GetThreeVector() {
		return fData;
	}

	double& operator[](int index) {
		return fData[index];
	}
	const double& operator[](const int index) const{
		return fData[index];
	}

	bool operator==( const KEMStreamableThreeVector& vector ) const {
		return(fData[0] == vector[0] && fData[1] == vector[1] && fData[2] == vector[2]);
	}

	bool operator!=( const KEMStreamableThreeVector& vector) const {
		return !(*this == vector);
	}

private:
	KThreeVector fData;
};


inline std::istream& operator>>( std::istream& aStream, KEMStreamableThreeVector& aVector )
{
    aStream >> aVector[ 0 ] >> aVector[ 1 ] >> aVector[ 2 ];
    return aStream;
}
inline std::ostream& operator<<( std::ostream& aStream, const KEMStreamableThreeVector& aVector )
{
    aStream << "<" << aVector[ 0 ] << " " << aVector[ 1 ] << " " << aVector[ 2 ] << ">";
    return aStream;
}

//
//template <typename InStream>
//InStream& operator>>(InStream& in,KEMStreamableThreeVector& vec) {
//	in >> vec[0] >> vec[1]  >> vec[2];
//	return in;
//}
//
//template <typename OutStream>
//OutStream& operator<<(OutStream& out,const KEMStreamableThreeVector& vec) {
//	out << vec[0] << vec[1] << vec[2];
//	return out;
//}

} //KEMField

#endif /* KEMSTREAMABLETHREEVECTOR_HH_ */
