/*
 * GblData.cpp
 *
 *  Created on: Aug 18, 2011
 *      Author: kleinwrt
 */

#include "GblData.h"

/// Create data block.
/**
 * \param [in] aLabel Label of corresponding point
 * \param [in] aValue Value of (scalar) measurement
 * \param [in] aPrec Precision of (scalar) measurement
 */
GblData::GblData(unsigned int aLabel, double aValue, double aPrec) :
		theLabel(aLabel), theValue(aValue), thePrecision(aPrec), theDownWeight(
				1.), thePrediction(0.), theParameters(), theDerivatives(), globalLabels(), globalDerivatives() {
	// TODO Auto-generated constructor stub
}

GblData::~GblData() {
	// TODO Auto-generated destructor stub
}

/// Add derivatives from measurement.
/**
 * Add (non-zero) derivatives to data block. Fill list of labels of used fit parameters.
 * \param [in] iRow Row index (0-4) in up to 5D measurement
 * \param [in] labDer Labels for derivatives
 * \param [in] matDer Derivatives (matrix) 'measurement vs track fit parameters'
 * \param [in] iOff Offset for row index for additional parameters
 * \param [in] derLocal Derivatives (matrix) for additional local parameters
 * \param [in] labGlobal Labels for additional global (MP-II) parameters
 * \param [in] derGlobal Derivatives (matrix) for additional global (MP-II) parameters
 */
void GblData::addDerivatives(unsigned int iRow,
		const std::vector<unsigned int> &labDer, const SMatrix55 &matDer,
		unsigned int iOff, const TMatrixD &derLocal,
		const std::vector<int> &labGlobal, const TMatrixD &derGlobal) {

	unsigned int nParMax = 5 + derLocal.GetNcols();
	theParameters.reserve(nParMax); // have to be sorted
	theDerivatives.reserve(nParMax);

	for (int i = 0; i < derLocal.GetNcols(); i++) // local derivatives
			{
		if (derLocal(iRow - iOff, i)) {
			theParameters.push_back(i + 1);
			theDerivatives.push_back(derLocal(iRow - iOff, i));
		}
	}

	for (unsigned int i = 0; i < 5; i++) // curvature, offset derivatives
			{
		if (labDer[i] and matDer(iRow, i)) {
			theParameters.push_back(labDer[i]);
			theDerivatives.push_back(matDer(iRow, i));
		}
	}

	globalLabels = labGlobal;
	for (int i = 0; i < derGlobal.GetNcols(); i++) // global derivatives
		globalDerivatives.push_back(derGlobal(iRow - iOff, i));
}

/// Add derivatives from kink.
/**
 * Add (non-zero) derivatives to data block. Fill list of labels of used fit parameters.
 * \param [in] iRow Row index (0-1) in 2D kink
 * \param [in] labDer Labels for derivatives
 * \param [in] matDer Derivatives (matrix) 'kink vs track fit parameters'
 */
void GblData::addDerivatives(unsigned int iRow,
		const std::vector<unsigned int> &labDer, const SMatrix27 &matDer) {

	unsigned int nParMax = 7;
	theParameters.reserve(nParMax); // have to be sorted
	theDerivatives.reserve(nParMax);

	for (unsigned int i = 0; i < 7; i++) // curvature, offset derivatives
			{
		if (labDer[i] and matDer(iRow, i)) {
			theParameters.push_back(labDer[i]);
			theDerivatives.push_back(matDer(iRow, i));
		}
	}
}

/// Add derivatives from external seed.
/**
 * Add (non-zero) derivatives to data block. Fill list of labels of used fit parameters.
 * \param [in] index Labels for derivatives
 * \param [in] derivatives Derivatives (vector)
 */
void GblData::addDerivatives(const std::vector<unsigned int> &index,
		const std::vector<double> &derivatives) {
	for (unsigned int i = 0; i < derivatives.size(); i++) // any derivatives
			{
		if (derivatives[i]) {
			theParameters.push_back(index[i]);
			theDerivatives.push_back(derivatives[i]);
		}
	}
}

/// Calculate compressed block matrix and right hand side from data.
/**
 * \param [out] anIndex List of labels of used fit parameters
 * \param [out] aVector Derivatives * Precision * Down-weighting
 * \param [out] aMatrix Derivatives * Precision * Down-weighting * Derivatives.T
 */
void GblData::getMatrices(std::vector<unsigned int> &anIndex, TVectorD &aVector,
		TMatrixDSym &aMatrix) {

	anIndex = theParameters;
	aVector.ResizeTo(theDerivatives.size());
	aMatrix.ResizeTo(theDerivatives.size(), theDerivatives.size());
	double aWeight = thePrecision * theDownWeight;
	for (unsigned int i = 0; i < theDerivatives.size(); i++) {
		aVector(i) = theDerivatives[i];
		for (unsigned int j = 0; j <= i; j++) {
			aMatrix(i, j) = aVector(i) * aVector(j) * aWeight;
		}
	}
	aVector *= (theValue * aWeight);
}

/// Calculate prediction for data from fit.
void GblData::setPrediction(const TVectorD &aVector) {

	thePrediction = 0.;
	for (unsigned int i = 0; i < theDerivatives.size(); i++) {
		thePrediction += theDerivatives[i] * aVector(theParameters[i] - 1);
	}
}

/// Outlier down weighting with M-estimators.
/**
 * \param [in] aMethod M-estimator (1: Tukey, 2:Huber, 3:Cauchy)
 */
double GblData::setDownWeighting(unsigned int aMethod) {

	double aWeight = 1.;
	double scaledResidual = fabs(theValue - thePrediction) * sqrt(thePrecision);
	if (aMethod == 1) // Tukey
			{
		if (scaledResidual < 4.6851) {
			aWeight = (1.0 - 0.045558 * scaledResidual * scaledResidual);
			aWeight *= aWeight;
		} else {
			aWeight = 0.;
		}
	} else if (aMethod == 2) //Huber
			{
		if (scaledResidual >= 1.345) {
			aWeight = 1.345 / scaledResidual;
		}
	} else if (aMethod == 3) //Cauchy
			{
		aWeight = 1.0 / (1.0 + (scaledResidual * scaledResidual / 5.6877));
	}
	theDownWeight = aWeight;
	return aWeight;
}

/// Calculate Chi2 contribution.
/**
 * \return (down-weighted) Chi2
 */
double GblData::getChi2() {
	double aDiff = theValue - thePrediction;
	return aDiff * aDiff * thePrecision * theDownWeight;
}

/// Print data block.
void GblData::printData() {

	std::cout << " meas. " << theLabel << "," << theValue << "," << thePrecision
			<< std::endl;
	std::cout << " param " << theParameters.size() << ":";
	for (unsigned int i = 0; i < theParameters.size(); i++) {
		std::cout << " " << theParameters[i];
	}
	std::cout << std::endl;
	std::cout << " deriv " << theDerivatives.size() << ":";
	for (unsigned int i = 0; i < theDerivatives.size(); i++) {
		std::cout << " " << theDerivatives[i];
	}
	std::cout << std::endl;
}

/// Get Data for local fit.
/**
 * \param [out] aValue Value
 * \param [out] aWeight Weight
 * \param [out] indLocal List of labels of used (local) fit parameters
 * \param [out] derLocal List of derivatives for used (local) fit parameters
 */
void GblData::getLocalData(double &aValue, double &aWeight,
		std::vector<unsigned int>* &indLocal, std::vector<double>* &derLocal) {

	aValue = theValue;
	aWeight = thePrecision * theDownWeight;
	indLocal = &theParameters;
	derLocal = &theDerivatives;
}

/// Get all Data for MP-II binary record.
/**
 * \param [out] fValue Value
 * \param [out] fErr Error
 * \param [out] indLocal List of labels of local parameters
 * \param [out] derLocal List of derivatives for local parameters
 * \param [out] labGlobal List of labels of global parameters
 * \param [out] derGlobal List of derivatives for global parameters
 */
void GblData::getAllData(float &fValue, float &fErr,
		std::vector<unsigned int>* &indLocal, std::vector<double>* &derLocal,
		std::vector<int>* &labGlobal, std::vector<double>* &derGlobal) {
	fValue = theValue;
	fErr = 1.0 / sqrt(thePrecision);
	indLocal = &theParameters;
	derLocal = &theDerivatives;
	labGlobal = &globalLabels;
	derGlobal = &globalDerivatives;
}