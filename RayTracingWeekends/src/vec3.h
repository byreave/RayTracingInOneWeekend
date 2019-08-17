//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <stdlib.h>
#include <iostream>

class vec3  {

    
public:
    vec3() {}
	vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; e[3] = 0.0f; }
	vec3(const __m128 & _e) :ee(_e) { }
    inline float x() const { return e[0]; }
    inline float y() const { return e[1]; }
    inline float z() const { return e[2]; }
    inline float r() const { return e[0]; }
    inline float g() const { return e[1]; }
    inline float b() const { return e[2]; }
    
    inline const vec3& operator+() const { return *this; }
    inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    inline float operator[](int i) const { return e[i]; }
    inline float& operator[](int i) { return e[i]; };
    
    inline vec3& operator+=(const vec3 &v2);
    inline vec3& operator-=(const vec3 &v2);
    inline vec3& operator*=(const vec3 &v2);
    inline vec3& operator/=(const vec3 &v2);
    inline vec3& operator*=(const float t);
    inline vec3& operator/=(const float t);
    
    inline float length() const { 
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(ee, ee, 0x71)));
		//return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); 
	}
	// 1/length() of the vector
	inline float rlength() const { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_dp_ps(ee, ee, 0x71))); }
    inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    inline void make_unit_vector();
    
	union {
		__m128 ee;
		float e[4];
	};
};



inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

inline void vec3::make_unit_vector() {
	ee =  _mm_mul_ps(ee, _mm_rsqrt_ps(_mm_dp_ps(ee, ee, 0x7F)));
    /*float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;*/
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
	return vec3(_mm_add_ps(v1.ee, v2.ee));
    //return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
	return vec3(_mm_sub_ps(v1.ee, v2.ee));
    //return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
	return vec3(_mm_mul_ps(v1.ee, v2.ee));
    //return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
	return vec3(_mm_div_ps(v1.ee, v2.ee));
    //return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline vec3 operator*(float t, const vec3 &v) {
	return vec3(_mm_mul_ps(v.ee, _mm_set1_ps(t)));
//  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator/(vec3 v, float t) {
	return vec3(_mm_div_ps(v.ee, _mm_set1_ps(t)));
    //return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

inline vec3 operator*(const vec3 &v, float t) {
	return vec3(_mm_mul_ps(v.ee, _mm_set1_ps(t)));
    //return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline float dot(const vec3 &v1, const vec3 &v2) {
	return _mm_cvtss_f32(_mm_dp_ps(v1.ee, v2.ee, 0x71));
    //return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    /*return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));*/
	return _mm_sub_ps(
		_mm_mul_ps(_mm_shuffle_ps(v1.ee, v1.ee, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(v2.ee, v2.ee, _MM_SHUFFLE(3, 1, 0, 2))),
		_mm_mul_ps(_mm_shuffle_ps(v1.ee, v1.ee, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(v2.ee, v2.ee, _MM_SHUFFLE(3, 0, 2, 1)))
	);
}


inline vec3& vec3::operator+=(const vec3 &v){
	/*e[0]  += v.e[0];
	e[1]  += v.e[1];
	e[2]  += v.e[2];*/
	ee = _mm_add_ps(ee, v.ee);
    return *this;
}

inline vec3& vec3::operator*=(const vec3 &v){
	/*e[0]  *= v.e[0];
	e[1]  *= v.e[1];
	e[2]  *= v.e[2];*/
	ee = _mm_mul_ps(ee, v.ee);
    return *this;
}

inline vec3& vec3::operator/=(const vec3 &v){
    /*e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];*/
	ee = _mm_div_ps(ee, v.ee);
    return *this;
}

inline vec3& vec3::operator-=(const vec3& v) {
    /*e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];*/
	ee = _mm_sub_ps(ee, v.ee);
    return *this;
}

inline vec3& vec3::operator*=(const float t) {
    /*e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;*/
	ee = _mm_mul_ps(ee, _mm_set1_ps(t));
    return *this;
}

inline vec3& vec3::operator/=(const float t) {
    float k = 1.0f/t;
    
    /*e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;*/
	ee = _mm_mul_ps(ee, _mm_set1_ps(k));
    return *this;
}

inline vec3 unit_vector(vec3 v) {
	return _mm_mul_ps(v.ee, _mm_rsqrt_ps(_mm_dp_ps(v.ee, v.ee, 0x7F)));
    //return v / v.length();
}

#endif
