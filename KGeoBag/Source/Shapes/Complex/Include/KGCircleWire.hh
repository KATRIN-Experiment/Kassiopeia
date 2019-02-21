#ifndef KGCIRCLEWIRE_DEF
#define KGCIRCLEWIRE_DEF

#include <stddef.h>
#include <vector>
#include <cmath>
#include <string>

#include "KGBoundary.hh"

namespace KGeoBag {

class KGCircleWire : public KGBoundary
{
	/*
	 A class describing a wire circle with a flat section profile
	 */
public:
	KGCircleWire() {
	}
	KGCircleWire(double r, double diameter, unsigned int nDisc) :
			fR(r), fDiameter(diameter), fNDisc(nDisc) {
	}

	virtual ~KGCircleWire() {
	}

	static std::string Name() {
		return "circle_wire";
	}

	virtual KGCircleWire* Clone() const;

	virtual void Initialize() const {
	}

	bool ContainsPoint(const double* P) const;
	double DistanceTo(const double* P, double* P_in = NULL, double* P_norm =
			NULL) const;

	double GetLength() const;
	double Area() const;
	double Volume() const;

	void SetR(double d) {
		fR = d;
	}
	void SetDiameter(double d) {
		fDiameter = d;
	}
	void SetNDisc(unsigned int d) {
		fNDisc = d;
	}

	double GetR() const {
		return fR;
	}
	double GetDiameter() const {
		return fDiameter;
	}
	unsigned int GetNDisc() const {
		return fNDisc;
	}

private:

	double fR;
	double fDiameter;
	unsigned int fNDisc;
};
}

#endif /* KGWIRECIRCLE_DEF */
