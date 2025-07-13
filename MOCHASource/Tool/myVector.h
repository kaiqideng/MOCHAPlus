#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#include <cmath>
#include <vector_functions.h>

HOST_DEVICE constexpr double pi()
{
	return 3.1415926535897931;
}

HOST_DEVICE inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_DEVICE inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_DEVICE inline double3 operator*(const double3& a, double s) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}

HOST_DEVICE inline double3 operator*(double s, const double3& a) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}

HOST_DEVICE inline double3 operator/(const double3& a, double s) {
    return make_double3(a.x / s, a.y / s, a.z / s);
}

HOST_DEVICE inline double3& operator+=(double3& a, const double3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

HOST_DEVICE inline double3& operator-=(double3& a, const double3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}
HOST_DEVICE inline double3& operator*=(double3& a, double s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}
HOST_DEVICE inline double3& operator/=(double3& a, double s) {
    a.x /= s; a.y /= s; a.z /= s;
    return a;
}

HOST_DEVICE inline double3 operator-(const double3& a) {
    return make_double3(-a.x, -a.y, -a.z);
}

HOST_DEVICE inline double dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HOST_DEVICE inline double3 cross(const double3& a, const double3& b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

HOST_DEVICE inline double lengthSquared(const double3& v) {
    return dot(v, v);
}

HOST_DEVICE inline double length(const double3& v) {
    return sqrt(dot(v, v));
}

HOST_DEVICE inline double3 normalize(const double3& v) {
    if (length(v) > 0) return v / sqrt(dot(v, v));
    else return v;
}

HOST_DEVICE inline double3 rotateVector(double3 v, double3 angleVector) {
	double angle_radians = length(angleVector);
	if (angle_radians < 1.e-10) return v; // No rotation needed

    // Rodrigues' rotation formula
    double3 k = angleVector / angle_radians; // Ensure axis is a unit vector
    double3 term1 = v * cos(angle_radians);
    double3 term2 = cross(k, v) * sin(angle_radians);
    double3 term3 = k * (dot(k, v) * (1 - cos(angle_radians)));

    return term1 + term2 + term3;
}

//HOST_DEVICE inline double3 rotateVector(double3 v, double3 axis, double angle_radians) {
//    double3 k = normalize(axis);  // Ensure axis is a unit vector
//
//    // Rodrigues' rotation formula
//    double3 term1 = v * cos(angle_radians);
//    double3 term2 = cross(k, v) * sin(angle_radians);
//    double3 term3 = k * (dot(k, v) * (1 - cos(angle_radians)));
//
//    return term1 + term2 + term3;
//}